#include <sys/time.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <assert.h>
#include <fcntl.h>
#include <errno.h>

#include <blitz/array.h>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_barrier.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>

#include "litmus.h"

using namespace blitz;
using namespace std;
using namespace boost::interprocess;

#define RESET_RELEASE_ON_MISS


void bail_out(const char* msg)
{
	perror(msg);
	exit(-1 * errno);
}


static void usage(char *error) {
	fprintf(stderr, "Error: %s\n", error);
	fprintf(stderr,
		"Usage:\n"
		"	rt_spin [COMMON-OPTS] WCET PERIOD DURATION\n"
		"	rt_spin [COMMON-OPTS] -f FILE [-o COLUMN] WCET PERIOD\n"
		"	rt_spin -l\n"
		"\n"
		"COMMON-OPTS = [-w] [-p PARTITION] [-c CLASS] [-s SCALE]\n"
		"\n"
		"WCET and PERIOD are milliseconds, DURATION is seconds.\n");
	exit(EXIT_FAILURE);
}

#define NUMS 4096
static int num[NUMS];

#define PAGE_SIZE (1024*4)

bool ENABLE_WAIT = true;
bool GPU_TASK = false;
bool ENABLE_AFFINITY = false;
bool USE_KFMLP = false;
bool RELAX_FIFO_MAX_LEN = false;
bool USE_DYNAMIC_GROUP_LOCKS = false;
bool BROADCAST_STATE = false;
bool ENABLE_CHUNKING = false;
bool MIGRATE_VIA_SYSMEM = false;
bool USE_PRIOQ = false;

int GPU_PARTITION = 0;
int GPU_PARTITION_SIZE = 0;
int NUM_SIMULT_USERS = 1;
size_t SEND_SIZE = 0;
size_t RECV_SIZE = 0;
size_t STATE_SIZE = 0;
size_t CHUNK_SIZE = PAGE_SIZE;


#define MAX_GPUS 8

int KEXCLU_LOCK;
int EE_LOCKS[MAX_GPUS];
int CE_SEND_LOCKS[MAX_GPUS];
int CE_RECV_LOCKS[MAX_GPUS];

int CUR_DEVICE = -1;
int LAST_DEVICE = -1;

bool useEngineLocks()
{
	return(NUM_SIMULT_USERS != 1);
}

int gpuCyclesPerSecond = 0;

uint64_t *init_release_time = NULL;
barrier *release_barrier = NULL;
barrier *gpu_barrier = NULL;
interprocess_mutex *gpu_mgmt_mutexes = NULL;
managed_shared_memory *segment_ptr = NULL;
managed_shared_memory *release_segment_ptr = NULL;

// observed average rate when four GPUs on same node in use from pagelocked memory.
// about 1/3 to 1/4 this when there is no bus contention.
//const double msPerByte = 4.22e-07;
//const double transOverhead = 0.01008;  // also observed.



char *d_send_data[MAX_GPUS] = {0};
char *d_recv_data[MAX_GPUS] = {0};
char *d_state_data[MAX_GPUS] = {0};

//cudaStream_t streams[MAX_GPUS];

char *h_send_data = 0;
char *h_recv_data = 0;
char *h_state_data = 0;


#include <sys/mman.h>
#define USE_PAGE_LOCKED_MEMORY
#ifdef USE_PAGE_LOCKED_MEMORY
#define c_malloc(s) \
		mmap(NULL, s ,   \
				PROT_READ | PROT_WRITE,  \
				MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED,  \
				-1, 0)
#else
#define c_malloc(s) malloc(s)
#endif

typedef int cudaError_t;
#define cudaSuccess 0

enum cudaMemcpyKind {
cudaMemcpyHostToDevice = 0,
cudaMemcpyDeviceToHost = 1,
cudaMemcpyDeviceToDevice = 2,
};

cudaError_t cudaGetLastError()
{
	return cudaSuccess;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

struct ce_lock_state
{
	int locks[2];
	size_t num_locks;
	size_t budget_remaining;
	bool locked;

	ce_lock_state(int device_a, enum cudaMemcpyKind kind, size_t size, int device_b = -1) {
		num_locks = (device_a != -1) + (device_b != -1);

		if(device_a != -1) {
			locks[0] = (kind == cudaMemcpyHostToDevice) ?
			CE_SEND_LOCKS[device_a] : CE_RECV_LOCKS[device_a];
		}

		if(device_b != -1) {
			assert(kind == cudaMemcpyDeviceToDevice);

			locks[1] = CE_RECV_LOCKS[device_b];

			if(locks[1] < locks[0]) {
				int temp = locks[1];
				locks[1] = locks[0];
				locks[0] = temp;
			}
		}

		if(!ENABLE_CHUNKING)
			budget_remaining = size;
		else
			budget_remaining = CHUNK_SIZE;
	}

	void lock() {
		if(USE_DYNAMIC_GROUP_LOCKS) {
			litmus_dgl_lock(locks, num_locks);
		}
		else
		{
			for(int l = 0; l < num_locks; ++l)
			{
				litmus_lock(locks[l]);
			}
		}
		locked = true;
	}

	void unlock() {
		if(USE_DYNAMIC_GROUP_LOCKS) {
			litmus_dgl_unlock(locks, num_locks);
		}
		else
		{
			// reverse order
			for(int l = num_locks - 1; l >= 0; --l)
			{
				litmus_unlock(locks[l]);
			}
		}
		locked = false;
	}

	void refresh() {
		budget_remaining = CHUNK_SIZE;
	}

	bool budgetIsAvailable(size_t tosend) {
		return(tosend >= budget_remaining);
	}

	void decreaseBudget(size_t spent) {
		budget_remaining -= spent;
	}
};

// precondition: if do_locking == true, locks in state are held.
cudaError_t __chunkMemcpy(void* a_dst, const void* a_src, size_t count,
						enum cudaMemcpyKind kind,
						ce_lock_state* state)
{
    cudaError_t ret = cudaSuccess;
    int remaining = count;

    char* dst = (char*)a_dst;
    const char* src = (const char*)a_src;

	// disable chunking, if needed, by setting chunk_size equal to the
	// amount of data to be copied.
	int chunk_size = (ENABLE_CHUNKING) ? CHUNK_SIZE : count;
	int i = 0;

    while(remaining != 0)
    {
        int bytesToCopy = std::min(remaining, chunk_size);

		if(state && state->budgetIsAvailable(bytesToCopy) && state->locked) {
			//cutilSafeCall( cudaStreamSynchronize(streams[CUR_DEVICE]) );
			ret = cudaGetLastError();

			if(ret != cudaSuccess)
			{
				break;
			}

			state->unlock();
			state->refresh(); // replentish.
							  // we can only run out of
							  // budget if chunking is enabled.
							  // we presume that init budget would
							  // be set to cover entire memcpy
							  // if chunking were disabled.
		}

		if(state && !state->locked) {
			state->lock();
		}

        //ret = cudaMemcpy(dst+i*chunk_size, src+i*chunk_size, bytesToCopy, kind);
		//cudaMemcpyAsync(dst+i*chunk_size, src+i*chunk_size, bytesToCopy, kind, streams[CUR_DEVICE]);

		if(state) {
			state->decreaseBudget(bytesToCopy);
		}

//		if(ret != cudaSuccess)
//		{
//			break;
//		}

        ++i;
        remaining -= bytesToCopy;
    }
    return ret;
}

cudaError_t chunkMemcpy(void* a_dst, const void* a_src, size_t count,
						enum cudaMemcpyKind kind,
						int device_a = -1,  // device_a == -1 disables locking
						bool do_locking = true,
						int device_b = -1)
{
	cudaError_t ret;
	if(!do_locking || device_a == -1) {
		ret = __chunkMemcpy(a_dst, a_src, count, kind, NULL);
		//cutilSafeCall( cudaStreamSynchronize(streams[CUR_DEVICE]) );
		if(ret == cudaSuccess)
			ret = cudaGetLastError();
	}
	else {
		ce_lock_state state(device_a, kind, count, device_b);
		state.lock();
		ret = __chunkMemcpy(a_dst, a_src, count, kind, &state);
		//cutilSafeCall( cudaStreamSynchronize(streams[CUR_DEVICE]) );
		if(ret == cudaSuccess)
			ret = cudaGetLastError();
		state.unlock();
	}
	return ret;
}


////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////


inline uint64_t timespec_to_ns(const struct timespec& t)
{
	return(t.tv_sec*1e9 + t.tv_nsec);
}

inline struct timespec ns_to_timespec(const uint64_t& ns)
{
	struct timespec temp = {ns/1e9, ns - ns/1e9};
	return(temp);
}

inline uint64_t clock_gettime_ns(clockid_t clk_id)
{
	struct timespec temp;
	clock_gettime(clk_id, &temp);
	return timespec_to_ns(temp);
}



static int loop_once(void)
{
	int i, j = 0;
	for (i = 0; i < NUMS; i++)
		j += num[i]++;
	return j;
}

static int loop_for(double exec_time, double emergency_exit)
{
	double last_loop = 0, loop_start;
	int tmp = 0;

	double start = cputime();
	double now = cputime();

	while (now + last_loop < start + exec_time) {
		loop_start = now;
		tmp += loop_once();
		now = cputime();
		last_loop = now - loop_start;
		if (emergency_exit && wctime() > emergency_exit) {
			/* Oops --- this should only be possible if the execution time tracking
			 * is broken in the LITMUS^RT kernel. */
			fprintf(stderr, "!!! rtspin/%d emergency exit!\n", getpid());
			fprintf(stderr, "Something is seriously wrong! Do not ignore this.\n");
			break;
		}
	}

	return tmp;
}

static void allocate_locks()
{
	// allocate k-FMLP lock
	int fd = open("semaphores", O_RDONLY | O_CREAT, S_IRUSR | S_IWUSR);

	int base_name = GPU_PARTITION * 1000;

	if(USE_KFMLP) {
		KEXCLU_LOCK = open_kfmlp_gpu_sem(fd,
										 base_name,  /* name */
										 GPU_PARTITION_SIZE,
										 GPU_PARTITION*GPU_PARTITION_SIZE,
										 NUM_SIMULT_USERS,
										 ENABLE_AFFINITY
										 );
	}
	else {
		KEXCLU_LOCK = open_gpusync_token_lock(fd,
										 base_name,  /* name */
										 GPU_PARTITION_SIZE,
										 GPU_PARTITION*GPU_PARTITION_SIZE,
										 NUM_SIMULT_USERS,
										 IKGLP_M_IN_FIFOS,
										 (!RELAX_FIFO_MAX_LEN) ?
											  IKGLP_OPTIMAL_FIFO_LEN :
											  IKGLP_UNLIMITED_FIFO_LEN,
										 ENABLE_AFFINITY
										 );
//		KEXCLU_LOCK = open_ikglp_gpu_sem(fd,
//										 base_name,  /* name */
//										 GPU_PARTITION_SIZE,
//										 GPU_PARTITION*GPU_PARTITION_SIZE,
//										 NUM_SIMULT_USERS,
//										 ENABLE_AFFINITY,
//										 RELAX_FIFO_MAX_LEN
//										 );
	}
	if(KEXCLU_LOCK < 0)
		perror("open_kexclu_sem");

	if(NUM_SIMULT_USERS > 1)
	{
		open_sem_t opensem = (!USE_PRIOQ) ? open_fifo_sem : open_prioq_sem;
		const char* opensem_label = (!USE_PRIOQ) ? "open_fifo_sem" : "open_prioq_sem";

		// allocate the engine locks.
		for (int i = 0; i < MAX_GPUS; ++i)
		{
			EE_LOCKS[i] = opensem(fd, (i+1)*10 + base_name);
			if(EE_LOCKS[i] < 0)
				perror(opensem_label);

			CE_SEND_LOCKS[i] = opensem(fd, (i+1)*10 + base_name + 1);
			if(CE_SEND_LOCKS[i] < 0)
				perror(opensem_label);

			if(NUM_SIMULT_USERS == 3)
			{
				// allocate a separate lock for the second copy engine
				CE_RECV_LOCKS[i] = opensem(fd, (i+1)*10 + base_name + 2);
				if(CE_RECV_LOCKS[i] < 0)
					perror(opensem_label);
			}
			else
			{
				// share a single lock for the single copy engine
				CE_RECV_LOCKS[i] = CE_SEND_LOCKS[i];
			}
		}
	}
}

static void allocate_host_memory()
{
	// round up to page boundaries
	size_t send_alloc_bytes = SEND_SIZE + (SEND_SIZE%PAGE_SIZE != 0)*PAGE_SIZE;
	size_t recv_alloc_bytes = RECV_SIZE + (RECV_SIZE%PAGE_SIZE != 0)*PAGE_SIZE;
	size_t state_alloc_bytes = STATE_SIZE + (STATE_SIZE%PAGE_SIZE != 0)*PAGE_SIZE;

	printf("Allocating host memory.  send = %dB, recv = %dB, state = %dB\n",
				send_alloc_bytes, recv_alloc_bytes, state_alloc_bytes);

//	if(send_alloc_bytes > 0)
//	{
//		h_send_data = (char *)c_malloc(send_alloc_bytes);
//		memset(h_send_data, 0x55, send_alloc_bytes);  // write some random value
//		// this will open a connection to GPU 0 if there is no active context, so
//		// expect long stalls.  LAME.
//		cutilSafeCall( cudaHostRegister(h_send_data, send_alloc_bytes, cudaHostRegisterPortable) );
//	}
//
//	if(recv_alloc_bytes > 0)
//	{
//		h_recv_data = (char *)c_malloc(recv_alloc_bytes);
//		memset(h_recv_data, 0xAA, recv_alloc_bytes);
//		cutilSafeCall( cudaHostRegister(h_recv_data, recv_alloc_bytes, cudaHostRegisterPortable) );
//	}
//
//	if(state_alloc_bytes > 0)
//	{
//		h_state_data = (char *)c_malloc(state_alloc_bytes);
//		memset(h_state_data, 0xCC, state_alloc_bytes);  // write some random value
//		cutilSafeCall( cudaHostRegister(h_state_data, state_alloc_bytes, cudaHostRegisterPortable) );
//	}

	printf("Host memory allocated.\n");
}

static void allocate_device_memory()
{
	printf("Allocating device memory.\n");
	// establish a connection to each GPU.
//	for(int i = 0; i < GPU_PARTITION_SIZE; ++i)
//	{
//		int which_device = GPU_PARTITION*GPU_PARTITION_SIZE + i;
//
//		if(ENABLE_WAIT) gpu_mgmt_mutexes[which_device].lock();
//
//		cutilSafeCall( cudaSetDevice(which_device) );
//		cutilSafeCall( cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 0) );
//		cutilSafeCall( cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0) );
//
//		cutilSafeCall( cudaStreamCreate(&streams[which_device]) );
//
//		/* pre-allocate memory, pray there's enough to go around */
//		if(SEND_SIZE > 0) {
//			cutilSafeCall( cudaMalloc((void**)&d_send_data[which_device], SEND_SIZE) );
//		}
//		if(RECV_SIZE > 0) {
//			cutilSafeCall( cudaMalloc((void**)&h_recv_data[which_device], RECV_SIZE) );
//		}
//		if(STATE_SIZE > 0) {
//			cutilSafeCall( cudaMalloc((void**)&h_state_data[which_device], STATE_SIZE) );
//		}
//
//		if(ENABLE_WAIT) gpu_mgmt_mutexes[which_device].unlock();
//	}
	printf("Device memory allocated.\n");
}

static void configure_gpus()
{
	printf("Configuring GPU\n");

//	// SUSPEND WHEN BLOCKED!!
//	cutilSafeCall( cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync) );
//
//	// establish a connection to each GPU.
//	for(int i = 0; i < GPU_PARTITION_SIZE; ++i)
//	{
//		int which_device = GPU_PARTITION*GPU_PARTITION_SIZE + i;
//
//		if(ENABLE_WAIT) gpu_mgmt_mutexes[which_device].lock();
//
//		cutilSafeCall( cudaSetDevice(which_device) );
//		cutilSafeCall( cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 0) );
//		cutilSafeCall( cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0) );
//
//		cutilSafeCall( cudaStreamCreate(&streams[which_device]) );
//
//		// enable P2P migrations.
//		// we assume all GPUs are on the same I/O hub.
//		for(int j = 0; j < GPU_PARTITION_SIZE; ++j)
//		{
//			int other_device = GPU_PARTITION*GPU_PARTITION_SIZE + j;
//
//			if(which_device != other_device)
//			{
//				cutilSafeCall( cudaDeviceEnablePeerAccess(other_device, 0) );
//			}
//		}
//
//		if(i == 0)
//		{
//			struct cudaDeviceProp pi;
//			cudaGetDeviceProperties(&pi, i);
//			gpuCyclesPerSecond = pi.clockRate * 1000; /* khz -> hz */
//		}
//
//		if(ENABLE_WAIT) gpu_mgmt_mutexes[which_device].unlock();
//	}

	printf("GPUs have been configured.\n");
}

static void init_cuda()
{
	configure_gpus();
	allocate_host_memory();
	allocate_device_memory();
	allocate_locks();
}

static void exit_cuda()
{
	for(int i = 0; i < GPU_PARTITION_SIZE; ++i)
	{
		int which_device = GPU_PARTITION*GPU_PARTITION_SIZE + i;

		if(ENABLE_WAIT) gpu_mgmt_mutexes[which_device].lock();

//		cutilSafeCall( cudaSetDevice(which_device) );
//		cutilSafeCall( cudaDeviceReset() );

		if(ENABLE_WAIT) gpu_mgmt_mutexes[which_device].unlock();
	}
}

static void catchExit(void)
{
	if(GPU_TASK)
	{
		// try to unlock everything.  litmus will prevent bogus calls.
		if(NUM_SIMULT_USERS > 1)
		{
			for(int i = 0; i < GPU_PARTITION_SIZE; ++i)
			{
				int which_device = GPU_PARTITION*GPU_PARTITION_SIZE + i;

				litmus_unlock(EE_LOCKS[which_device]);
				litmus_unlock(CE_SEND_LOCKS[which_device]);
				if(NUM_SIMULT_USERS == 2) {
					litmus_unlock(CE_RECV_LOCKS[which_device]);
				}
			}
		}

		if(CUR_DEVICE >= 0) {
			unregister_nv_device(CUR_DEVICE);
		}

		litmus_unlock(KEXCLU_LOCK);
	}
}

static void migrateToGPU(int destination)
{
	if(!BROADCAST_STATE && STATE_SIZE > 0)
	{
		if(MIGRATE_VIA_SYSMEM)
		{
			chunkMemcpy(h_state_data, d_state_data[LAST_DEVICE], STATE_SIZE,
						cudaMemcpyDeviceToHost, LAST_DEVICE, useEngineLocks());
		}
	}

//	cutilSafeCall( cudaSetDevice(destination) );

	if(!BROADCAST_STATE && STATE_SIZE > 0)
	{
		if(MIGRATE_VIA_SYSMEM)
		{
			chunkMemcpy(d_state_data[CUR_DEVICE], h_state_data, STATE_SIZE,
						cudaMemcpyHostToDevice, CUR_DEVICE, useEngineLocks());
		}
		else
		{
			chunkMemcpy(d_state_data[destination],
						d_state_data[LAST_DEVICE],
						STATE_SIZE,
						cudaMemcpyDeviceToDevice,
						CUR_DEVICE,
						useEngineLocks(),
						destination);
		}
	}
}

static void broadcastState(int from)
{
	if(STATE_SIZE > 0)
	{
		assert(CUR_DEVICE == from);

		if(MIGRATE_VIA_SYSMEM)
		{
			chunkMemcpy(h_state_data, d_state_data[from], STATE_SIZE,
						cudaMemcpyDeviceToHost, from, useEngineLocks());
		}

		for(int i = 0; i < GPU_PARTITION_SIZE; ++i)
		{
			int which_device = GPU_PARTITION*GPU_PARTITION_SIZE + i;
			if(which_device != from)
			{
				if(MIGRATE_VIA_SYSMEM)
				{
//					cutilSafeCall( cudaSetDevice(which_device) );
					CUR_DEVICE = which_device; // temporary
					chunkMemcpy(d_state_data[which_device], h_state_data, STATE_SIZE,
								cudaMemcpyHostToDevice, which_device, useEngineLocks());
				}
				else
				{
					chunkMemcpy(d_state_data[which_device],
								d_state_data[from],
								STATE_SIZE,
								cudaMemcpyDeviceToDevice,
								from,
								useEngineLocks(),
								which_device);
				}
			}
		}

		if(MIGRATE_VIA_SYSMEM && CUR_DEVICE != from)
		{
//			cutilSafeCall( cudaSetDevice(from) );
			CUR_DEVICE = from;
		}
	}
}

//// Executes on graphics card.
//__global__ void docudaspin(unsigned int cycles)
//{
//	long long unsigned int elapsed = 0;
//	long long int now = clock64();
//	long long int last;
//	do
//	{
//		last = now;
//		now = clock64();
//		elapsed += max(0ll, (long long int)(now - last)); // don't count iterations with clock roll-over
//	}while(elapsed < cycles);
//
//	return;
//}



static void gpu_loop_for(double gpu_sec_time, double emergency_exit)
{
	unsigned int numcycles = (unsigned int)(gpuCyclesPerSecond * gpu_sec_time);
	int numblocks = 1;
	int blocksz = 1;

	CUR_DEVICE = litmus_lock(KEXCLU_LOCK);
	{
		if(CUR_DEVICE != LAST_DEVICE && LAST_DEVICE != -1)
		{
			migrateToGPU(CUR_DEVICE);
		}

		if(SEND_SIZE > 0)
		{
			// handles chunking and locking, as appropriate.
			chunkMemcpy(d_send_data[CUR_DEVICE], h_send_data, SEND_SIZE,
						cudaMemcpyHostToDevice, CUR_DEVICE, useEngineLocks());
		}

		if(useEngineLocks()) litmus_lock(EE_LOCKS[CUR_DEVICE]);

//		docudaspin <<<numblocks,blocksz, 0, streams[CUR_DEVICE]>>> (numcycles);
//		cutilSafeCall( cudaStreamSynchronize(streams[CUR_DEVICE]) );

		if(useEngineLocks()) litmus_unlock(EE_LOCKS[CUR_DEVICE]);

		if(RECV_SIZE > 0)
		{
			chunkMemcpy(h_recv_data, d_recv_data[CUR_DEVICE], RECV_SIZE,
						cudaMemcpyDeviceToHost, CUR_DEVICE, useEngineLocks());
		}

		if(BROADCAST_STATE)
		{
			broadcastState(CUR_DEVICE);
		}
	}
	litmus_unlock(KEXCLU_LOCK);

	LAST_DEVICE = CUR_DEVICE;
	CUR_DEVICE = -1;
}


static void debug_delay_loop(void)
{
	double start, end, delay;

	while (1) {
		for (delay = 0.5; delay > 0.01; delay -= 0.01) {
			start = wctime();
			loop_for(delay, 0);
			end = wctime();
			printf("%6.4fs: looped for %10.8fs, delta=%11.8fs, error=%7.4f%%\n",
			       delay,
			       end - start,
			       end - start - delay,
			       100 * (end - start - delay) / delay);
		}
	}
}

static int job(double exec_time, double gpu_sec_time, double program_end)
{
	if (wctime() > program_end)
		return 0;
	else if (!GPU_TASK)
	{
		loop_for(exec_time, program_end + 1);
	}
	else
	{
		double cpu_bookend = (exec_time)/2.0;

		loop_for(cpu_bookend, program_end + 1);
		gpu_loop_for(gpu_sec_time, program_end + 1);
		loop_for(cpu_bookend, program_end + 1);
	}
	return 1;
}

#define OPTSTR "p:ls:e:g:G:W:N:S:R:T:BMaLyC:rz:q"

int main(int argc, char** argv)
{
	atexit(catchExit);

	int ret;
	lt_t wcet;
	lt_t period;
	double wcet_ms, period_ms;
	int migrate = 0;
	int cpu = 0;
	int opt;
	int test_loop = 0;
//	int column = 1;
	const char *file = NULL;
	int want_enforcement = 0;
	double duration = 0, releaseTime = 0;
	double *exec_times = NULL;
	double scale = 1.0;
	uint64_t cur_job;
	uint64_t num_jobs;

	int create_shm = -1;
	int num_tasks = 0;

	double gpu_sec_ms = 0;

	while ((opt = getopt(argc, argv, OPTSTR)) != -1) {
//		printf("opt = %c optarg = %s\n", opt, optarg);
		switch (opt) {
//		case 'w':
//			ENABLE_WAIT = 1;
//			break;
		case 'p':
			cpu = atoi(optarg);
			migrate = 1;
			break;
		case 'l':
			test_loop = 1;
			break;
		case 's':
			scale = atof(optarg);
			break;
		case 'e':
			gpu_sec_ms = atof(optarg);
			break;
//		case 'x':
//			trans_sec_ms = atof(optarg);
//			break;
		case 'z':
			NUM_SIMULT_USERS = atoi(optarg);
			break;
		case 'q':
			USE_PRIOQ = true;
			break;
		case 'g':
			GPU_TASK = 1;
			GPU_PARTITION_SIZE = atoi(optarg);
			break;
		case 'G':
			GPU_PARTITION = atoi(optarg);
			break;
		case 'S':
			SEND_SIZE = (size_t)(atof(optarg)*1024);
			break;
		case 'R':
			RECV_SIZE = (size_t)(atof(optarg)*1024);
			break;
		case 'T':
			STATE_SIZE = (size_t)(atof(optarg)*1024);
			break;
		case 'B':
			BROADCAST_STATE = true;
			break;
		case 'M':
			MIGRATE_VIA_SYSMEM = true;
			break;
		case 'a':
			ENABLE_AFFINITY = true;
			break;
		case 'r':
			RELAX_FIFO_MAX_LEN = true;
			break;
		case 'L':
			USE_KFMLP = true;
			break;
		case 'y':
			USE_DYNAMIC_GROUP_LOCKS = true;
			break;
		case 'C':
			ENABLE_CHUNKING = true;
			CHUNK_SIZE = (size_t)(atof(optarg)*1024);
			break;
		case 'W':
			create_shm = atoi(optarg);
			break;
		case 'N':
			num_tasks = atoi(optarg);
			break;
		case ':':
			usage("Argument missing.");
			break;
		case '?':
		default:
			usage("Bad argument.");
			break;
		}
	}

	if (test_loop) {
		debug_delay_loop();
		return 0;
	}

//	if (file) {
//		int num_jobs_tmp;
//		get_exec_times(file, column, &num_jobs_tmp, &exec_times);
//		num_jobs = num_jobs_tmp;
//
//		if (argc - optind < 2)
//			usage("Arguments missing.");
//
//		for (cur_job = 0; cur_job < num_jobs; ++cur_job) {
//			/* convert the execution time to seconds */
//			duration += exec_times[cur_job] * 0.001;
//		}
//	} else {
		/*
		 * if we're not reading from the CSV file, then we need
		 * three parameters
		 */
		if (argc - optind < 3)
			usage("Arguments missing.");
//	}

	wcet_ms   = atof(argv[optind + 0]);
	period_ms = atof(argv[optind + 1]);

	wcet   = wcet_ms * __NS_PER_MS;
	period = period_ms * __NS_PER_MS;
	if (wcet <= 0)
		usage("The worst-case execution time must be a "
				"positive number.");
	if (period <= 0)
		usage("The period must be a positive number.");
	if (!file && wcet > period) {
		usage("The worst-case execution time must not "
				"exceed the period.");
	}

	if (!file)
	{
		duration  = atof(argv[optind + 2]);
		num_jobs = ((double)duration*1e3)/period_ms;
		++num_jobs; // padding
	}
	else if (file && num_jobs > 1)
	{
		duration += period_ms * 0.001 * (num_jobs - 1);
	}

	if (migrate) {
		ret = be_migrate_to(cpu);
		if (ret < 0)
			bail_out("could not migrate to target partition");
	}

	if(ENABLE_WAIT)
	{
		if(num_tasks > 0)
		{
			printf("%d creating release shared memory\n", getpid());
			shared_memory_object::remove("release_barrier_memory");
			release_segment_ptr = new managed_shared_memory(create_only, "release_barrier_memory", 4*1024);

			printf("%d creating release barrier for %d users\n", getpid(), num_tasks);
			release_barrier = release_segment_ptr->construct<barrier>("barrier release_barrier")(num_tasks);

			init_release_time = release_segment_ptr->construct<uint64_t>("uint64_t instance")();
			*init_release_time = 0;
		}
		else
		{
			do
			{
				try
				{
					printf("%d opening release shared memory\n", getpid());
					segment_ptr = new managed_shared_memory(open_only, "release_barrier_memory");
				}
				catch(...)
				{
					printf("%d shared memory not ready.  sleeping\n", getpid());
					sleep(1);
				}
			}while(segment_ptr == NULL);

			release_barrier = segment_ptr->find<barrier>("barrier release_barrier").first;
			init_release_time = segment_ptr->find<uint64_t>("uint64_t instance").first;
		}
	}


	if(GPU_TASK)
	{
		if(ENABLE_WAIT)
		{
			if(create_shm > -1)
			{
				printf("%d creating shared memory\n", getpid());
				shared_memory_object::remove("gpu_barrier_memory");
			 	segment_ptr = new managed_shared_memory(create_only, "gpu_barrier_memory", 4*1024);

				printf("%d creating a barrier for %d users\n", getpid(), create_shm);
			 	gpu_barrier = segment_ptr->construct<barrier>("barrier instance")(create_shm);
				printf("%d creating gpu mgmt mutexes for 8 devices\n", getpid());
				gpu_mgmt_mutexes = segment_ptr->construct<interprocess_mutex>("interprocess_mutex m")[8]();
			}
			else
			{
				do
				{
					try
					{
						printf("%d opening shared memory\n", getpid());
						segment_ptr = new managed_shared_memory(open_only, "gpu_barrier_memory");
					}
					catch(...)
					{
						printf("%d shared memory not ready.  sleeping\n", getpid());
						sleep(1);
					}
				}while(segment_ptr == NULL);

				gpu_barrier = segment_ptr->find<barrier>("barrier instance").first;
				gpu_mgmt_mutexes = segment_ptr->find<interprocess_mutex>("interprocess_mutex m").first;
			}
		}

		// scale data transmission too??
		SEND_SIZE *= scale;
		RECV_SIZE *= scale;
		STATE_SIZE *= scale;

		init_cuda();
	}

	ret = sporadic_task_ns(wcet, period, 0, cpu, RT_CLASS_SOFT,
			       want_enforcement ? PRECISE_ENFORCEMENT
			                        : NO_ENFORCEMENT,
			       migrate);
	if (ret < 0)
		bail_out("could not setup rt task params");

	init_litmus();

	ret = task_mode(LITMUS_RT_TASK);
	if (ret != 0)
		bail_out("could not become RT task");



	uint64_t jobCount = 0;
	blitz::Array<uint64_t, 1> responseTimeLog(num_jobs+1);

	struct timespec spec;
	uint64_t release;
	uint64_t finish;


	if (ENABLE_WAIT) {
		printf("Waiting for release.\n");
		ret = wait_for_ts_release();
		if (ret != 0)
			bail_out("wait_for_ts_release()");
	}
	else
	{
		sleep_next_period();
	}

	clock_gettime(CLOCK_MONOTONIC, &spec);
	release = timespec_to_ns(spec);
	if (!__sync_bool_compare_and_swap(init_release_time, 0, release))
	{
		release = *init_release_time;
	}

	releaseTime = wctime();
	double failsafeEnd = releaseTime + duration;


	if (file) {
		/* use times read from the CSV file */
		for (cur_job = 0; cur_job < num_jobs; ++cur_job) {
			/* convert job's length to seconds */
			job(exec_times[cur_job] * 0.001 * scale,
					gpu_sec_ms * 0.001 * scale,
					failsafeEnd);
		}
	} else {
		/* convert to seconds and scale */
		int keepGoing;
		do
		{
			keepGoing = job(wcet_ms * 0.001 * scale, gpu_sec_ms * 0.001 * scale, failsafeEnd);


			clock_gettime(CLOCK_MONOTONIC, &spec);
			finish = timespec_to_ns(spec);

			responseTimeLog(min(num_jobs,jobCount++)) = finish - release;

			// this is an estimated upper-bound on release time.  it may be off by several microseconds.
#ifdef RESET_RELEASE_ON_MISS
			release = (release + period < finish) ?
					finish :  /* missed deadline.  adopt next release as current time. */
					release + period;  /* some time in the future. */
#else
			release = release + period; // allow things to get progressively later.
#endif

			sleep_next_period();
			clock_gettime(CLOCK_MONOTONIC, &spec);
			release = min(timespec_to_ns(spec), release);

		} while(keepGoing);
	}

	if(GPU_TASK && ENABLE_WAIT)
	{
		printf("%d waiting at barrier\n", getpid());
		gpu_barrier->wait();
	}

	ret = task_mode(BACKGROUND_TASK);
	if (ret != 0)
		bail_out("could not become regular task (huh?)");

	if (file)
		free(exec_times);

	if(GPU_TASK)
	{
		/*
		if(ENABLE_WAIT)
		{
			// wait for all GPU using tasks ext RT mode.
			printf("%d waiting at barrier\n", getpid());
			gpu_barrier->wait();
		}
		*/

		exit_cuda();

		if(ENABLE_WAIT)
		{
			/* wait before we clean up memory */
			printf("%d waiting for all to shutdown GPUs\n", getpid());
			gpu_barrier->wait();

/*
			if(create_shm > -1)
			{
				printf("%d removing shared memory\n", getpid());
				shared_memory_object::remove("gpu_barrier_memory");
			}
*/
		}
	}


	if (ENABLE_WAIT)
	{
		printf("%d waiting at exit barrier\n", getpid());
		release_barrier->wait();
	}


	char gpu_using_str[] = "GPU\n";
	char cpu_only_str[] = "CPU\n";
	#define USED(arr) (arr)(Range(fromStart,min(num_jobs-1,jobCount-1)))
	// period (ms), avg-rt, min-rt, max-rt, avg-slack, numMisses
	printf("DONE,%d,%d,%f,%f,%f,%lu,%lu,%f,%lu,%d,%d,%s",
		   cpu,
		   getpid(),
		   period_ms,
		   // average
		   blitz::mean(USED(responseTimeLog)),
		   // average pct of period
		   100.0*(blitz::mean(USED(responseTimeLog))/period),
		   // min
		   blitz::min(USED(responseTimeLog)),
		   // max
		   blitz::max(USED(responseTimeLog)),
		   // average slack
		   blitz::mean((uint64_t)period - USED(responseTimeLog)),
		   // num jobs
		   min(num_jobs-1,jobCount-1),
		   // num misses
		   blitz::count(USED(responseTimeLog) > (uint64_t)period),
		   // num misses w/ unbounded
		   blitz::count(USED(responseTimeLog) > (uint64_t)(2*period)),
		   // flag gpu-using tasks
		   ((GPU_TASK) ? gpu_using_str : cpu_only_str)
		   );

	return 0;
}
