#include <sys/time.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <execinfo.h>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>

#include <random/normal.h>

#include <cuda_runtime.h>

#include "litmus.h"
#include "common.h"

using namespace std;
using namespace boost::interprocess;
using namespace ranlib;

#define ms2s(ms)  ((ms)*0.001)

const char *lock_namespace = "./.gpuspin-locks";

const int NR_GPUS = 8;

bool GPU_USING = false;
bool ENABLE_AFFINITY = false;
bool RELAX_FIFO_MAX_LEN = false;
bool ENABLE_CHUNKING = false;
bool MIGRATE_VIA_SYSMEM = false;

enum eEngineLockTypes
{
	FIFO,
	PRIOQ
};

eEngineLockTypes ENGINE_LOCK_TYPE = FIFO;

int GPU_PARTITION = 0;
int GPU_PARTITION_SIZE = 0;
int CPU_PARTITION_SIZE = 0;

int RHO = 2;

int NUM_COPY_ENGINES = 2;


__attribute__((unused)) static size_t kbToB(size_t kb) { return kb * 1024; }
__attribute__((unused)) static size_t mbToB(size_t mb) { return kbToB(mb * 1024); }

/* in bytes */
size_t SEND_SIZE = 0;
size_t RECV_SIZE = 0;
size_t STATE_SIZE = 0;
size_t CHUNK_SIZE = 0;

int TOKEN_LOCK = -1;

bool USE_ENGINE_LOCKS = false;
bool USE_DYNAMIC_GROUP_LOCKS = false;
int EE_LOCKS[NR_GPUS];
int CE_SEND_LOCKS[NR_GPUS];
int CE_RECV_LOCKS[NR_GPUS];
int CE_MIGR_SEND_LOCKS[NR_GPUS];
int CE_MIGR_RECV_LOCKS[NR_GPUS];
bool RESERVED_MIGR_COPY_ENGINE = false;  // only checked if NUM_COPY_ENGINES == 2

//bool ENABLE_RT_AUX_THREADS = false;
bool ENABLE_RT_AUX_THREADS = true;

enum eGpuSyncMode
{
	IKGLP_MODE,
	IKGLP_WC_MODE, /* work-conserving IKGLP. no GPU is left idle, but breaks optimality */
	KFMLP_MODE,
	RGEM_MODE,
};

eGpuSyncMode GPU_SYNC_MODE = IKGLP_MODE;

enum eCudaSyncMode
{
	BLOCKING,
	SPIN
};

eCudaSyncMode CUDA_SYNC_MODE = BLOCKING;


int CUR_DEVICE = -1;
int LAST_DEVICE = -1;

cudaStream_t STREAMS[NR_GPUS];
int GPU_HZ[NR_GPUS];
int NUM_SM[NR_GPUS];
int WARP_SIZE[NR_GPUS];
int ELEM_PER_THREAD[NR_GPUS];

#define DEFINE_PER_GPU(type, var) type var[NR_GPUS]
#define per_gpu(var, idx) (var[(idx)])
#define this_gpu(var) (var[(CUR_DEVICE)])
#define cur_stream() (this_gpu(STREAMS))
#define cur_gpu() (CUR_DEVICE)
#define last_gpu() (LAST_DEVICE)
#define cur_ee() (EE_LOCKS[CUR_DEVICE])
#define cur_send() (CE_SEND_LOCKS[CUR_DEVICE])
#define cur_recv() (CE_RECV_LOCKS[CUR_DEVICE])
#define cur_migr_send() (CE_MIGR_SEND_LOCKS[CUR_DEVICE])
#define cur_migr_recv() (CE_MIGR_RECV_LOCKS[CUR_DEVICE])
#define cur_hz() (GPU_HZ[CUR_DEVICE])
#define cur_sms() (NUM_SM[CUR_DEVICE])
#define cur_warp_size() (WARP_SIZE[CUR_DEVICE])
#define cur_elem_per_thread() (ELEM_PER_THREAD[CUR_DEVICE])
#define num_online_gpus() (NUM_GPUS)

static bool useEngineLocks()
{
	return(USE_ENGINE_LOCKS);
}

//#define VANILLA_LINUX

bool TRACE_MIGRATIONS = false;
#ifndef VANILLA_LINUX
#define trace_migration(to, from)					do { inject_gpu_migration((to), (from)); } while(0)
#define trace_release(arrival, deadline, jobno)		do { inject_release((arrival), (deadline), (jobno)); } while(0)
#define trace_completion(jobno)						do { inject_completion((jobno)); } while(0)
#define trace_name()								do { inject_name(); } while(0)
#define trace_param()								do { inject_param(); } while(0)
#else
#define set_rt_task_param(x, y)						(0)
#define trace_migration(to, from)
#define trace_release(arrival, deadline, jobno)
#define trace_completion(jobno)
#define trace_name()
#define trace_param()
#endif

struct ce_lock_state
{
	int locks[2];
	size_t num_locks;
	size_t budget_remaining;
	bool locked;

	ce_lock_state(int device_a, enum cudaMemcpyKind kind, size_t size, int device_b = -1, bool migration = false) {
		num_locks = (device_a != -1) + (device_b != -1);

		if(device_a != -1) {
			if (!migration)
				locks[0] = (kind == cudaMemcpyHostToDevice || (kind == cudaMemcpyDeviceToDevice && device_b == -1)) ?
				CE_SEND_LOCKS[device_a] : CE_RECV_LOCKS[device_a];
			else
				locks[0] = (kind == cudaMemcpyHostToDevice || (kind == cudaMemcpyDeviceToDevice && device_b == -1)) ?
				CE_MIGR_SEND_LOCKS[device_a] : CE_MIGR_RECV_LOCKS[device_a];
		}

		if(device_b != -1) {
			assert(kind == cudaMemcpyDeviceToDevice);

			if (!migration)
				locks[1] = CE_RECV_LOCKS[device_b];
			else
				locks[1] = CE_MIGR_RECV_LOCKS[device_b];

			if(locks[1] < locks[0]) {
				// enforce total order on locking
				int temp = locks[1];
				locks[1] = locks[0];
				locks[0] = temp;
			}
		}
		else {
			locks[1] = -1;
		}

		if(!ENABLE_CHUNKING)
			budget_remaining = size;
		else
			budget_remaining = CHUNK_SIZE;
	}

	void crash(void) {
		void *array[50];
		int size, i;
		char **messages;

		size = backtrace(array, 50);
		messages = backtrace_symbols(array, size);

		fprintf(stderr, "%d: TRIED TO GRAB SAME LOCK TWICE! Lock = %d\n", getpid(), locks[0]);
		for (i = 1; i < size && messages != NULL; ++i)
		{
			fprintf(stderr, "%d: [bt]: (%d) %s\n", getpid(), i, messages[i]);
		}
		free(messages);

		assert(false);
	}


	void lock() {
		if(locks[0] == locks[1]) crash();

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
		if(locks[0] == locks[1]) crash();

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
static cudaError_t __chunkMemcpy(void* a_dst, const void* a_src, size_t count,
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
			cudaStreamSynchronize(STREAMS[CUR_DEVICE]);
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
		cudaMemcpyAsync(dst+i*chunk_size, src+i*chunk_size, bytesToCopy, kind, STREAMS[CUR_DEVICE]);

		if(state) {
			state->decreaseBudget(bytesToCopy);
		}

        ++i;
        remaining -= bytesToCopy;
    }
    return ret;
}

static cudaError_t chunkMemcpy(void* a_dst, const void* a_src, size_t count,
							   enum cudaMemcpyKind kind,
							   int device_a = -1,  // device_a == -1 disables locking
							   bool do_locking = true,
							   int device_b = -1,
							   bool migration = false)
{
	cudaError_t ret;
	if(!do_locking || device_a == -1) {
		ret = __chunkMemcpy(a_dst, a_src, count, kind, NULL);
		cudaStreamSynchronize(cur_stream());
		if(ret == cudaSuccess)
			ret = cudaGetLastError();
	}
	else {
		ce_lock_state state(device_a, kind, count, device_b, migration);
		state.lock();
		ret = __chunkMemcpy(a_dst, a_src, count, kind, &state);
		cudaStreamSynchronize(cur_stream());
		if(ret == cudaSuccess)
			ret = cudaGetLastError();
		state.unlock();
	}
	return ret;
}


void allocate_locks_litmus(void)
{
	// allocate k-FMLP lock
	int fd = open(lock_namespace, O_RDONLY | O_CREAT, S_IRUSR | S_IWUSR);

	int base_name = GPU_PARTITION * 1000;

	if (GPU_SYNC_MODE == IKGLP_MODE) {
		/* Standard (optimal) IKGLP */
		TOKEN_LOCK = open_gpusync_token_lock(fd,
						base_name,  /* name */
						GPU_PARTITION_SIZE,
						GPU_PARTITION*GPU_PARTITION_SIZE,
						RHO,
						IKGLP_M_IN_FIFOS,
						(!RELAX_FIFO_MAX_LEN) ?
						IKGLP_OPTIMAL_FIFO_LEN :
						IKGLP_UNLIMITED_FIFO_LEN,
						ENABLE_AFFINITY);
	}
	else if (GPU_SYNC_MODE == KFMLP_MODE) {
		/* KFMLP. FIFO queues only for tokens. */
		TOKEN_LOCK = open_gpusync_token_lock(fd,
						base_name,  /* name */
						GPU_PARTITION_SIZE,
						GPU_PARTITION*GPU_PARTITION_SIZE,
						RHO,
						IKGLP_UNLIMITED_IN_FIFOS,
						IKGLP_UNLIMITED_FIFO_LEN,
						ENABLE_AFFINITY);
	}
	else if (GPU_SYNC_MODE == RGEM_MODE) {
		/* RGEM-like token allocation. Shared priority queue for all tokens. */
		TOKEN_LOCK = open_gpusync_token_lock(fd,
						base_name,  /* name */
						GPU_PARTITION_SIZE,
						GPU_PARTITION*GPU_PARTITION_SIZE,
						RHO,
						RHO*GPU_PARTITION_SIZE,
						1,
						ENABLE_AFFINITY);
	}
	else if (GPU_SYNC_MODE == IKGLP_WC_MODE) {
		/* Non-optimal IKGLP that never lets a replica idle if there are pending
		 * token requests. */
		int max_simult_run = std::max(CPU_PARTITION_SIZE, RHO*GPU_PARTITION_SIZE);
		int max_fifo_len = (int)ceil((float)max_simult_run / (RHO*GPU_PARTITION_SIZE));
		TOKEN_LOCK = open_gpusync_token_lock(fd,
						base_name,  /* name */
						GPU_PARTITION_SIZE,
						GPU_PARTITION*GPU_PARTITION_SIZE,
						RHO,
						max_simult_run,
						(!RELAX_FIFO_MAX_LEN) ?
							max_fifo_len :
							IKGLP_UNLIMITED_FIFO_LEN,
						ENABLE_AFFINITY);
	}
	else {
		perror("Invalid GPUSync mode specified\n");
		TOKEN_LOCK = -1;
	}

	if(TOKEN_LOCK < 0)
		perror("open_token_sem");

	if(USE_ENGINE_LOCKS)
	{
		assert(NUM_COPY_ENGINES == 1 || NUM_COPY_ENGINES == 2);
		assert((NUM_COPY_ENGINES == 1 && !RESERVED_MIGR_COPY_ENGINE) || NUM_COPY_ENGINES == 2);

		// allocate the engine locks.
		for (int i = 0; i < GPU_PARTITION_SIZE; ++i)
		{
			int idx = GPU_PARTITION*GPU_PARTITION_SIZE + i;
			int ee_name = (i+1)*10 + base_name;
			int ce_0_name = (i+1)*10 + base_name + 1;
			int ce_1_name = (i+1)*10 + base_name + 2;
			int ee_lock = -1, ce_0_lock = -1, ce_1_lock = -1;

			open_sem_t openEngineLock = (ENGINE_LOCK_TYPE == FIFO) ?
				open_fifo_sem : open_prioq_sem;

			ee_lock = openEngineLock(fd, ee_name);
			if (ee_lock < 0)
				perror("open_*_sem (engine lock)");

			ce_0_lock = openEngineLock(fd, ce_0_name);
			if (ce_0_lock < 0)
				perror("open_*_sem (engine lock)");

			if (NUM_COPY_ENGINES == 2)
			{
				ce_1_lock = openEngineLock(fd, ce_1_name);
				if (ce_1_lock < 0)
					perror("open_*_sem (engine lock)");
			}

			EE_LOCKS[idx] = ee_lock;

			if (NUM_COPY_ENGINES == 1)
			{
				// share locks
				CE_SEND_LOCKS[idx] = ce_0_lock;
				CE_RECV_LOCKS[idx] = ce_0_lock;
				CE_MIGR_SEND_LOCKS[idx] = ce_0_lock;
				CE_MIGR_RECV_LOCKS[idx] = ce_0_lock;
			}
			else
			{
				assert(NUM_COPY_ENGINES == 2);

				if (RESERVED_MIGR_COPY_ENGINE) {
					// copy engine deadicated to migration operations
					CE_SEND_LOCKS[idx] = ce_0_lock;
					CE_RECV_LOCKS[idx] = ce_0_lock;
					CE_MIGR_SEND_LOCKS[idx] = ce_1_lock;
					CE_MIGR_RECV_LOCKS[idx] = ce_1_lock;
				}
				else {
					// migration transmissions treated as regular data
					CE_SEND_LOCKS[idx] = ce_0_lock;
					CE_RECV_LOCKS[idx] = ce_1_lock;
					CE_MIGR_SEND_LOCKS[idx] = ce_0_lock;
					CE_MIGR_RECV_LOCKS[idx] = ce_1_lock;
				}
			}
		}
	}
}




class gpu_pool
{
public:
    gpu_pool(int pSz): poolSize(pSz)
    {
		memset(&pool[0], 0, sizeof(pool[0])*poolSize);
    }

    int get(pthread_mutex_t* tex, int preference = -1)
    {
        int which = -1;
	//	int last = (preference >= 0) ? preference : 0;
		int last = (ENABLE_AFFINITY) ?
				(preference >= 0) ? preference : 0 :
				rand()%poolSize;
		int minIdx = last;

		pthread_mutex_lock(tex);

		int min = pool[last];
		for(int i = (minIdx+1)%poolSize; i != last; i = (i+1)%poolSize)
		{
			if(min > pool[i])
				minIdx = i;
		}
		++pool[minIdx];

		pthread_mutex_unlock(tex);

		which = minIdx;

        return which;
    }

    void put(pthread_mutex_t* tex, int which)
    {
		pthread_mutex_lock(tex);
		--pool[which];
		pthread_mutex_unlock(tex);
    }

private:
	int poolSize;
    int pool[NR_GPUS]; // >= gpu_part_size
};

static gpu_pool* GPU_LINUX_SEM_POOL = NULL;
static pthread_mutex_t* GPU_LINUX_MUTEX_POOL = NULL;

static void allocate_locks_linux(const int num_gpu_users)
{
	managed_shared_memory *segment_pool_ptr = NULL;
	managed_shared_memory *segment_mutex_ptr = NULL;

	int numGpuPartitions = NR_GPUS/GPU_PARTITION_SIZE;

	if(num_gpu_users > 0)
	{
		printf("%d creating shared memory for linux semaphores; num pools = %d, pool size = %d\n", getpid(), numGpuPartitions, GPU_PARTITION_SIZE);
		shared_memory_object::remove("linux_mutex_memory");
		shared_memory_object::remove("linux_sem_memory");

		segment_mutex_ptr = new managed_shared_memory(create_only, "linux_mutex_memory", 4*1024);
		GPU_LINUX_MUTEX_POOL = segment_mutex_ptr->construct<pthread_mutex_t>("pthread_mutex_t linux_m")[numGpuPartitions]();
		for(int i = 0; i < numGpuPartitions; ++i)
		{
			pthread_mutexattr_t attr;
			pthread_mutexattr_init(&attr);
			pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
			pthread_mutex_init(&(GPU_LINUX_MUTEX_POOL[i]), &attr);
			pthread_mutexattr_destroy(&attr);
		}

		segment_pool_ptr = new managed_shared_memory(create_only, "linux_sem_memory", 4*1024);
		GPU_LINUX_SEM_POOL = segment_pool_ptr->construct<gpu_pool>("gpu_pool linux_p")[numGpuPartitions](GPU_PARTITION_SIZE);
	}
	else
	{
		do
		{
			try
			{
				if (!segment_pool_ptr) segment_pool_ptr = new managed_shared_memory(open_only, "linux_sem_memory");
			}
			catch(...)
			{
				sleep(1);
			}
		}while(segment_pool_ptr == NULL);

		do
		{
			try
			{
				if (!segment_mutex_ptr) segment_mutex_ptr = new managed_shared_memory(open_only, "linux_mutex_memory");
			}
			catch(...)
			{
				sleep(1);
			}
		}while(segment_mutex_ptr == NULL);

		GPU_LINUX_SEM_POOL = segment_pool_ptr->find<gpu_pool>("gpu_pool linux_p").first;
		GPU_LINUX_MUTEX_POOL = segment_mutex_ptr->find<pthread_mutex_t>("pthread_mutex_t linux_m").first;
	}
}




static void allocate_locks(const int num_gpu_users, bool linux_mode)
{
	if(!linux_mode)
		allocate_locks_litmus();
	else
		allocate_locks_linux(num_gpu_users);
}

static void set_cur_gpu(int gpu)
{
	if (TRACE_MIGRATIONS) {
		trace_migration(gpu, CUR_DEVICE);
	}
	if(gpu != CUR_DEVICE) {
		cudaSetDevice(gpu);
		CUR_DEVICE = gpu;
	}
}


static pthread_barrier_t *gpu_barrier = NULL;
static interprocess_mutex *gpu_mgmt_mutexes = NULL;
static managed_shared_memory *segment_ptr = NULL;

void coordinate_gpu_tasks(const int num_gpu_users)
{
	if(num_gpu_users > 0)
	{
		printf("%d creating shared memory\n", getpid());
		shared_memory_object::remove("gpu_barrier_memory");
		segment_ptr = new managed_shared_memory(create_only, "gpu_barrier_memory", 4*1024);

		printf("%d creating a barrier for %d users\n", getpid(), num_gpu_users);
		gpu_barrier = segment_ptr->construct<pthread_barrier_t>("pthread_barrier_t gpu_barrier")();
		pthread_barrierattr_t battr;
		pthread_barrierattr_init(&battr);
		pthread_barrierattr_setpshared(&battr, PTHREAD_PROCESS_SHARED);
		pthread_barrier_init(gpu_barrier, &battr, num_gpu_users);
		pthread_barrierattr_destroy(&battr);
		printf("%d creating gpu mgmt mutexes for %d devices\n", getpid(), NR_GPUS);
		gpu_mgmt_mutexes = segment_ptr->construct<interprocess_mutex>("interprocess_mutex m")[NR_GPUS]();
	}
	else
	{
		do
		{
			try
			{
				segment_ptr = new managed_shared_memory(open_only, "gpu_barrier_memory");
			}
			catch(...)
			{
				sleep(1);
			}
		}while(segment_ptr == NULL);

		gpu_barrier = segment_ptr->find<pthread_barrier_t>("pthread_barrier_t gpu_barrier").first;
		gpu_mgmt_mutexes = segment_ptr->find<interprocess_mutex>("interprocess_mutex m").first;
	}
}

typedef float spindata_t;

char *d_send_data[NR_GPUS] = {0};
char *d_recv_data[NR_GPUS] = {0};
char *d_state_data[NR_GPUS] = {0};
spindata_t *d_spin_data[NR_GPUS] = {0};
//unsigned int *d_iteration_count[NR_GPUS] = {0};


bool p2pMigration[NR_GPUS][NR_GPUS] = {0};

char *h_send_data = 0;
char *h_recv_data = 0;
char *h_state_data = 0;

unsigned int *h_iteration_count[NR_GPUS] = {0};

static void init_cuda(const int num_gpu_users)
{
	const int PAGE_SIZE = 4*1024;
	size_t send_alloc_bytes = SEND_SIZE + (SEND_SIZE%PAGE_SIZE != 0)*PAGE_SIZE;
	size_t recv_alloc_bytes = RECV_SIZE + (RECV_SIZE%PAGE_SIZE != 0)*PAGE_SIZE;
	size_t state_alloc_bytes = STATE_SIZE + (STATE_SIZE%PAGE_SIZE != 0)*PAGE_SIZE;

	coordinate_gpu_tasks(num_gpu_users);

#if 1
	switch (CUDA_SYNC_MODE)
	{
		case BLOCKING:
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			break;
		case SPIN:
			cudaSetDeviceFlags(cudaDeviceScheduleSpin);
			break;
	}
#else
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
#endif

	for(int i = 0; i < GPU_PARTITION_SIZE; ++i)
	{
		cudaDeviceProp prop;
		int which = GPU_PARTITION*GPU_PARTITION_SIZE + i;

		gpu_mgmt_mutexes[which].lock();
		try
		{
			set_cur_gpu(which);
			cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 0);
			cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0);

			cudaGetDeviceProperties(&prop, which);
			GPU_HZ[which] = prop.clockRate * 1000; /* khz -> hz */
			NUM_SM[which] = prop.multiProcessorCount;
			WARP_SIZE[which] = prop.warpSize;

			// enough to fill the L2 cache exactly.
			ELEM_PER_THREAD[which] = (prop.l2CacheSize/(NUM_SM[which]*WARP_SIZE[which]*sizeof(spindata_t)));


			if (!MIGRATE_VIA_SYSMEM && prop.unifiedAddressing)
			{
				for(int j = 0; j < GPU_PARTITION_SIZE; ++j)
				{
					if (i != j)
					{
						int other = GPU_PARTITION*GPU_PARTITION_SIZE + j;
						int canAccess = 0;
						cudaDeviceCanAccessPeer(&canAccess, which, other);
						if(canAccess)
						{
							cudaDeviceEnablePeerAccess(other, 0);
							p2pMigration[which][other] = true;
						}
					}
				}
			}

			cudaStreamCreate(&STREAMS[CUR_DEVICE]);

			cudaMalloc(&d_spin_data[which], prop.l2CacheSize);
			cudaMemset(&d_spin_data[which], 0, prop.l2CacheSize);
//			cudaMalloc(&d_iteration_count[which], NUM_SM[which]*WARP_SIZE[which]*sizeof(unsigned int));
//			cudaHostAlloc(&h_iteration_count[which], NUM_SM[which]*WARP_SIZE[which]*sizeof(unsigned int), cudaHostAllocPortable | cudaHostAllocMapped);

			if (send_alloc_bytes) {
				cudaMalloc(&d_send_data[which], send_alloc_bytes);
				cudaHostAlloc(&h_send_data, send_alloc_bytes, cudaHostAllocPortable | cudaHostAllocMapped);
			}

			if (h_recv_data) {
				cudaMalloc(&d_recv_data[which], recv_alloc_bytes);
				cudaHostAlloc(&h_recv_data, recv_alloc_bytes, cudaHostAllocPortable | cudaHostAllocMapped);
			}

			if (h_state_data) {
				cudaMalloc(&d_state_data[which], state_alloc_bytes);

				if (MIGRATE_VIA_SYSMEM)
					cudaHostAlloc(&h_state_data, state_alloc_bytes, cudaHostAllocPortable | cudaHostAllocMapped | cudaHostAllocWriteCombined);
			}
		}
		catch(std::exception &e)
		{
			printf("caught an exception during initializiation!: %s\n", e.what());
		}
		catch(...)
		{
			printf("caught unknown exception.\n");
		}

		gpu_mgmt_mutexes[which].unlock();
	}

	// roll back to first GPU
	set_cur_gpu(GPU_PARTITION*GPU_PARTITION_SIZE);
}



static bool MigrateToGPU_P2P(int from, int to)
{
	bool success = true;
	set_cur_gpu(to);
	chunkMemcpy(this_gpu(d_state_data), per_gpu(d_state_data, from),
				STATE_SIZE, cudaMemcpyDeviceToDevice, to,
				useEngineLocks(), from, true);
	return success;
}


static bool PullState(void)
{
	bool success = true;
	chunkMemcpy(h_state_data, this_gpu(d_state_data),
				STATE_SIZE, cudaMemcpyDeviceToHost,
				cur_gpu(), useEngineLocks(), -1, true);
	return success;
}

static bool PushState(void)
{
	bool success = true;
	chunkMemcpy(this_gpu(d_state_data), h_state_data,
				STATE_SIZE, cudaMemcpyHostToDevice,
				cur_gpu(), useEngineLocks(), -1, true);
	return success;
}

static bool MigrateToGPU_SysMem(int from, int to)
{
	// THIS IS ON-DEMAND SYS_MEM MIGRATION.  GPUSync says
	// you should be using speculative migrations.
	// Use PushState() and PullState().
	assert(false); // for now

	bool success = true;

	set_cur_gpu(from);
	chunkMemcpy(h_state_data, this_gpu(d_state_data),
				STATE_SIZE, cudaMemcpyDeviceToHost,
				from, useEngineLocks(), -1, true);

	set_cur_gpu(to);
	chunkMemcpy(this_gpu(d_state_data), h_state_data,
				STATE_SIZE, cudaMemcpyHostToDevice,
				to, useEngineLocks(), -1, true);

	return success;
}

static bool MigrateToGPU(int from, int to)
{
	bool success = false;

	if (from != to)
	{
		if(!MIGRATE_VIA_SYSMEM && p2pMigration[to][from])
			success = MigrateToGPU_P2P(from, to);
		else
			success = MigrateToGPU_SysMem(from, to);
	}
	else
	{
		set_cur_gpu(to);
		success = true;
	}

	return success;
}

static bool MigrateToGPU_Implicit(int to)
{
	return( MigrateToGPU(cur_gpu(), to) );
}

static void MigrateIfNeeded(int next_gpu)
{
	if(next_gpu != cur_gpu() && cur_gpu() != -1)
	{
		if (!MIGRATE_VIA_SYSMEM)
			MigrateToGPU_Implicit(next_gpu);
		else {
			set_cur_gpu(next_gpu);
			PushState();
		}
	}
}



static void exit_cuda()
{
	for(int i = 0; i < GPU_PARTITION_SIZE; ++i)
	{
		int which = GPU_PARTITION*GPU_PARTITION_SIZE + i;
		gpu_mgmt_mutexes[which].lock();
		set_cur_gpu(which);
		cudaDeviceReset();
		gpu_mgmt_mutexes[which].unlock();
	}
}

bool safetynet = false;

static void catch_exit(int catch_exit)
{
	if(GPU_USING && USE_ENGINE_LOCKS && safetynet)
	{
		safetynet = false;
		for(int i = 0; i < GPU_PARTITION_SIZE; ++i)
		{
			int which = GPU_PARTITION*GPU_PARTITION_SIZE + i;
			set_cur_gpu(which);

//			cudaDeviceReset();

			// try to unlock everything.  litmus will prevent bogus calls.
			if(USE_ENGINE_LOCKS)
			{
				litmus_unlock(EE_LOCKS[which]);
				litmus_unlock(CE_SEND_LOCKS[which]);
				if (NUM_COPY_ENGINES == 2)
				{
					if (RESERVED_MIGR_COPY_ENGINE)
						litmus_unlock(CE_MIGR_SEND_LOCKS[which]);
					else
						litmus_unlock(CE_MIGR_RECV_LOCKS[which]);
				}
			}
		}
		litmus_unlock(TOKEN_LOCK);
	}
}





#ifdef VANILLA_LINUX
static float ms_sum;
static int gpucount = 0;
#endif

__global__ void docudaspin(float* data, /*unsigned int* iterations,*/ unsigned int num_elem, unsigned int cycles)
{
	long long int now = clock64();
	long long unsigned int elapsed = 0;
	long long int last;

//	unsigned int iter = 0;
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = 0;
	bool toggle = true;

//	iterations[i] = 0;
	do
	{
		data[i*num_elem+j] += (toggle) ? M_PI : -M_PI;
		j = (j + 1 != num_elem) ? j + 1 : 0;
		toggle = !toggle;
//		iter++;

		last = now;
		now = clock64();

//		// exact calculation takes more cycles than a second
//		// loop iteration when code is compiled optimized
//		long long int diff = now - last;
//		elapsed += (diff > 0) ?
//			diff :
//			now + ((~((long long int)0)<<1)>>1) - last;

		// don't count iterations with clock roll-over
		elapsed += max(0ll, now - last);
	}while(elapsed < cycles);

//	iterations[i] = iter;

	return;
}

static void gpu_loop_for(double gpu_sec_time, unsigned int num_kernels, double emergency_exit)
{
	int next_gpu;

	if (gpu_sec_time <= 0.0)
		goto out;
	if (emergency_exit && wctime() > emergency_exit)
		goto out;

	next_gpu = litmus_lock(TOKEN_LOCK);
	{
		MigrateIfNeeded(next_gpu);
		unsigned int numcycles = ((unsigned int)(cur_hz() * gpu_sec_time))/num_kernels;

		if(SEND_SIZE > 0)
			chunkMemcpy(this_gpu(d_state_data), h_send_data, SEND_SIZE,
						cudaMemcpyHostToDevice, CUR_DEVICE, useEngineLocks());

		for(unsigned int i = 0; i < num_kernels; ++i)
		{
			if(useEngineLocks()) litmus_lock(cur_ee());
			/* one block per sm, one warp per block */
			docudaspin <<<cur_sms(),cur_warp_size(), 0, cur_stream()>>> (d_spin_data[cur_gpu()], cur_elem_per_thread(), numcycles);
			cudaStreamSynchronize(cur_stream());
			if(useEngineLocks()) litmus_unlock(cur_ee());
		}

		if(RECV_SIZE > 0)
			chunkMemcpy(h_recv_data, this_gpu(d_state_data), RECV_SIZE,
						cudaMemcpyDeviceToHost, CUR_DEVICE, useEngineLocks());

		if (MIGRATE_VIA_SYSMEM)
			PullState();
	}
	litmus_unlock(TOKEN_LOCK);

	last_gpu() = cur_gpu();

out:
	return;
}

static void gpu_loop_for_linux(double gpu_sec_time, unsigned int num_kernels, double emergency_exit)
{
	static int GPU_OFFSET = GPU_PARTITION * GPU_PARTITION_SIZE;
	static gpu_pool *pool = &GPU_LINUX_SEM_POOL[GPU_PARTITION];
	static pthread_mutex_t *mutex = &GPU_LINUX_MUTEX_POOL[GPU_PARTITION];

	int next_gpu;

	if (gpu_sec_time <= 0.0)
		goto out;
	if (emergency_exit && wctime() > emergency_exit)
		goto out;

#ifdef VANILLA_LINUX
	static bool once = false;
	static cudaEvent_t start, end;
	float ms;
	if (!once)
	{
		once = true;
		cudaEventCreate(&start);
		cudaEventCreate(&end);
	}
#endif

	next_gpu = pool->get(mutex, cur_gpu() - GPU_OFFSET) + GPU_OFFSET;
	{
		MigrateIfNeeded(next_gpu);

		unsigned int numcycles = ((unsigned int)(cur_hz() * gpu_sec_time))/num_kernels;

		if(SEND_SIZE > 0)
			chunkMemcpy(this_gpu(d_state_data), h_send_data, SEND_SIZE,
						cudaMemcpyHostToDevice, cur_gpu(), useEngineLocks());

		for(unsigned int i = 0; i < num_kernels; ++i)
		{
			/* one block per sm, one warp per block */
#ifdef VANILLA_LINUX
			cudaEventRecord(start, cur_stream());
#endif
			docudaspin <<<cur_sms(),cur_warp_size(), 0, cur_stream()>>> (d_spin_data[cur_gpu()], cur_elem_per_thread(), numcycles);
#ifdef VANILLA_LINUX
			cudaEventRecord(end, cur_stream());
			cudaEventSynchronize(end);
#endif
			cudaStreamSynchronize(cur_stream());

#ifdef VANILLA_LINUX
			cudaEventElapsedTime(&ms, start, end);
			ms_sum += ms;
#endif
		}
#ifdef VANILLA_LINUX
		++gpucount;
#endif

		if(RECV_SIZE > 0)
			chunkMemcpy(h_recv_data, this_gpu(d_state_data), RECV_SIZE,
						cudaMemcpyDeviceToHost, cur_gpu(), useEngineLocks());

		if (MIGRATE_VIA_SYSMEM)
			PullState();
	}
	pool->put(mutex, cur_gpu() - GPU_OFFSET);

	last_gpu() = cur_gpu();

out:
	return;
}




static void usage(char *error) {
	fprintf(stderr, "Error: %s\n", error);
	fprintf(stderr,
		"Usage:\n"
		"	rt_spin [COMMON-OPTS] WCET PERIOD DURATION\n"
		"	rt_spin [COMMON-OPTS] -f FILE [-o COLUMN] WCET PERIOD\n"
		"	rt_spin -l\n"
		"\n"
		"COMMON-OPTS = [-w] [-s SCALE]\n"
		"              [-p PARTITION/CLUSTER [-z CLUSTER SIZE]] [-c CLASS]\n"
		"              [-X LOCKING-PROTOCOL] [-L CRITICAL SECTION LENGTH] [-Q RESOURCE-ID]"
		"\n"
		"WCET and PERIOD are milliseconds, DURATION is seconds.\n"
		"CRITICAL SECTION LENGTH is in milliseconds.\n");
	exit(EXIT_FAILURE);
}

/*
 * returns the character that made processing stop, newline or EOF
 */
static int skip_to_next_line(FILE *fstream)
{
	int ch;
	for (ch = fgetc(fstream); ch != EOF && ch != '\n'; ch = fgetc(fstream));
	return ch;
}

static void skip_comments(FILE *fstream)
{
	int ch;
	for (ch = fgetc(fstream); ch == '#'; ch = fgetc(fstream))
		skip_to_next_line(fstream);
	ungetc(ch, fstream);
}

static void get_exec_times(const char *file, const int column,
			   int *num_jobs,    double **exec_times)
{
	FILE *fstream;
	int  cur_job, cur_col, ch;
	*num_jobs = 0;

	fstream = fopen(file, "r");
	if (!fstream)
		bail_out("could not open execution time file");

	/* figure out the number of jobs */
	do {
		skip_comments(fstream);
		ch = skip_to_next_line(fstream);
		if (ch != EOF)
			++(*num_jobs);
	} while (ch != EOF);

	if (-1 == fseek(fstream, 0L, SEEK_SET))
		bail_out("rewinding file failed");

	/* allocate space for exec times */
	*exec_times = (double*)calloc(*num_jobs, sizeof(*exec_times));
	if (!*exec_times)
		bail_out("couldn't allocate memory");

	for (cur_job = 0; cur_job < *num_jobs && !feof(fstream); ++cur_job) {

		skip_comments(fstream);

		for (cur_col = 1; cur_col < column; ++cur_col) {
			/* discard input until we get to the column we want */
			int unused __attribute__ ((unused)) = fscanf(fstream, "%*s,");
		}

		/* get the desired exec. time */
		if (1 != fscanf(fstream, "%lf", (*exec_times)+cur_job)) {
			fprintf(stderr, "invalid execution time near line %d\n",
					cur_job);
			exit(EXIT_FAILURE);
		}

		skip_to_next_line(fstream);
	}

	assert(cur_job == *num_jobs);
	fclose(fstream);
}

#define NUMS 4096
static int num[NUMS];
__attribute__((unused)) static char* progname;

static int loop_once(void)
{
	int i, j = 0;
	for (i = 0; i < NUMS; i++)
		j += num[i]++;
	return j;
}

static int loop_for(double exec_time, double emergency_exit)
{
	int tmp = 0;
	double last_loop, loop_start;
	double start, now;

	if (exec_time <= 0.0)
		goto out;

	start = cputime();
	now = cputime();

	if (emergency_exit && wctime() > emergency_exit)
		goto out;

	last_loop = 0;
	while (now + last_loop < start + exec_time) {
		loop_start = now;
		tmp += loop_once();
		now = cputime();
		last_loop = now - loop_start;
		if (emergency_exit && wctime() > emergency_exit) {
			/* Oops --- this should only be possible if the execution time tracking
			 * is broken in the LITMUS^RT kernel. */
			fprintf(stderr, "!!! gpuspin/%d emergency exit!\n", getpid());
			fprintf(stderr, "Something is seriously wrong! Do not ignore this.\n");
			break;
		}
	}

out:
	return tmp;
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

typedef bool (*gpu_job_t)(double exec_time, double gpu_exec_time, unsigned int num_kernels, double program_end);
typedef bool (*cpu_job_t)(double exec_time, double program_end);

static bool gpu_job(double exec_time, double gpu_exec_time, unsigned int num_kernels, double program_end)
{
	double chunk1, chunk2;

	if (wctime() > program_end) {
		return false;
	}
	else {
		chunk1 = exec_time * drand48();
		chunk2 = exec_time - chunk1;

		loop_for(chunk1, program_end + 1);
		gpu_loop_for(gpu_exec_time, num_kernels, program_end + 1);
		loop_for(chunk2, program_end + 1);

		sleep_next_period();
	}
	return true;
}

static bool job(double exec_time, double program_end)
{
	if (wctime() > program_end) {
		return false;
	}
	else {
		loop_for(exec_time, program_end + 1);
		sleep_next_period();
	}
	return true;
}

/*****************************/
/* only used for linux modes */

static struct timespec periodTime;
static struct timespec releaseTime;
static unsigned int job_no = 0;

static lt_t period_ns;

static void log_release()
{
	__attribute__ ((unused)) lt_t rel = releaseTime.tv_sec * s2ns(1) + releaseTime.tv_nsec;
	__attribute__ ((unused)) lt_t dead = rel + period_ns;
	trace_release(rel, dead, job_no);
}

static void log_completion()
{
	trace_completion(job_no);
	++job_no;
}

static void setup_next_period_linux(struct timespec* spec, struct timespec* period)
{
	spec->tv_sec += period->tv_sec;
	spec->tv_nsec += period->tv_nsec;
	if (spec->tv_nsec >= s2ns(1)) {
		++(spec->tv_sec);
		spec->tv_nsec -= s2ns(1);
	}
}

static void sleep_next_period_linux()
{
	log_completion();
	setup_next_period_linux(&releaseTime, &periodTime);
	clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &releaseTime, NULL);
	log_release();
}

static void init_linux()
{
	mlockall(MCL_CURRENT | MCL_FUTURE);
}

static bool gpu_job_linux(double exec_time, double gpu_exec_time, unsigned int num_kernels, double program_end)
{
	double chunk1, chunk2;

	if (wctime() > program_end) {
		return false;
	}
	else {
		chunk1 = exec_time * drand48();
		chunk2 = exec_time - chunk1;

		loop_for(chunk1, program_end + 1);
		gpu_loop_for_linux(gpu_exec_time, num_kernels, program_end + 1);
		loop_for(chunk2, program_end + 1);

		sleep_next_period_linux();
	}
	return true;
}

static bool job_linux(double exec_time, double program_end)
{
	if (wctime() > program_end) {
		return false;
	}
	else {
		loop_for(exec_time, program_end + 1);
		sleep_next_period_linux();
	}
	return true;
}

/*****************************/

enum eScheduler
{
	LITMUS,
	LINUX,
	RT_LINUX
};

#define CPU_OPTIONS "p:z:c:wlveio:f:s:q:X:L:Q:d:"
#define GPU_OPTIONS "g:y:r:C:E:DG:xS:R:T:Z:aFm:b:MNIk:VW:"

// concat the option strings
#define OPTSTR CPU_OPTIONS GPU_OPTIONS

int main(int argc, char** argv)
{
	int ret;

	struct rt_task param;

	lt_t wcet;
	lt_t period;
	lt_t budget;
	double wcet_ms = -1.0;
	double gpu_wcet_ms = 0.0;
	double period_ms = -1.0;
	double budget_ms = -1.0;

	unsigned int num_kernels = 1;

	budget_drain_policy_t drain = DRAIN_SIMPLE;
	bool want_enforcement = false;
	bool want_signals = false;

	unsigned int priority = LITMUS_LOWEST_PRIORITY;

	task_class_t cls = RT_CLASS_SOFT;

	eScheduler scheduler = LITMUS;
	int num_gpu_users = 0;
	int migrate = 0;
	int cluster = 0;
	int cluster_size = 1;

	Normal<double> *wcet_dist_ms = NULL;
	float stdpct = 0.0;

	cpu_job_t cjobfn = NULL;
	gpu_job_t gjobfn = NULL;

	int wait = 0;
	double scale = 1.0;
	int test_loop = 0;

	double duration = 0, start = 0;
	int cur_job = 0, num_jobs = 0;
	int column = 1;

	int opt;

	double *exec_times = NULL;
	const char *file = NULL;

	/* locking */
//	int lock_od = -1;
//	int resource_id = 0;
//	int protocol = -1;
//	double cs_length = 1; /* millisecond */

	progname = argv[0];

	while ((opt = getopt(argc, argv, OPTSTR)) != -1) {
		switch (opt) {
		case 'w':
			wait = 1;
			break;
		case 'p':
			cluster = atoi(optarg);
			migrate = 1;
			break;
		case 'z':
			cluster_size = atoi(optarg);
			CPU_PARTITION_SIZE = cluster_size;
			break;
		case 'g':
			GPU_USING = true;
			GPU_PARTITION = atoi(optarg);
			assert(GPU_PARTITION >= 0 && GPU_PARTITION < NR_GPUS);
			break;
		case 'y':
			GPU_PARTITION_SIZE = atoi(optarg);
			assert(GPU_PARTITION_SIZE > 0);
			break;
		case 'r':
			RHO = atoi(optarg);
			assert(RHO > 0);
			break;
		case 'C':
			NUM_COPY_ENGINES = atoi(optarg);
			assert(NUM_COPY_ENGINES == 1 || NUM_COPY_ENGINES == 2);
			break;
		case 'V':
			RESERVED_MIGR_COPY_ENGINE = true;
			break;
		case 'E':
			USE_ENGINE_LOCKS = true;
			ENGINE_LOCK_TYPE = (eEngineLockTypes)atoi(optarg);
			assert(ENGINE_LOCK_TYPE == FIFO || ENGINE_LOCK_TYPE == PRIOQ);
			break;
		case 'D':
			USE_DYNAMIC_GROUP_LOCKS = true;
			break;
		case 'G':
			GPU_SYNC_MODE = (eGpuSyncMode)atoi(optarg);
			assert(GPU_SYNC_MODE >= IKGLP_MODE && GPU_SYNC_MODE <= RGEM_MODE);
			break;
		case 'a':
			ENABLE_AFFINITY = true;
			break;
		case 'F':
			RELAX_FIFO_MAX_LEN = true;
			break;
		case 'x':
			CUDA_SYNC_MODE = SPIN;
			break;
		case 'S':
			SEND_SIZE = kbToB((size_t)atoi(optarg));
			break;
		case 'R':
			RECV_SIZE = kbToB((size_t)atoi(optarg));
			break;
		case 'T':
			STATE_SIZE = kbToB((size_t)atoi(optarg));
			break;
		case 'Z':
			ENABLE_CHUNKING = true;
			CHUNK_SIZE = kbToB((size_t)atoi(optarg));
			break;
		case 'M':
			MIGRATE_VIA_SYSMEM = true;
			break;
		case 'm':
			num_gpu_users = (int)atoi(optarg);
			assert(num_gpu_users > 0);
			break;
		case 'k':
			num_kernels = (unsigned int)atoi(optarg);
			break;
		case 'b':
			budget_ms = atoi(optarg);
			break;
		case 'W':
			stdpct = atof(optarg);
			break;
		case 'N':
			scheduler = LINUX;
			break;
		case 'I':
			scheduler = RT_LINUX;
			break;
		case 'q':
			priority = atoi(optarg);
			break;
		case 'c':
			cls = str2class(optarg);
			if (cls == -1)
				usage("Unknown task class.");
			break;
		case 'e':
			want_enforcement = true;
			break;
		case 'i':
			want_signals = true;
			break;
		case 'd':
			drain = (budget_drain_policy_t)atoi(optarg);
			assert(drain >= DRAIN_SIMPLE && drain <= DRAIN_SOBLIV);
			assert(drain != DRAIN_SAWARE); // unsupported
			break;
		case 'l':
			test_loop = 1;
			break;
		case 'o':
			column = atoi(optarg);
			break;
//		case 'f':
//			file = optarg;
//			break;
		case 's':
			scale = atof(optarg);
			break;
//		case 'X':
//			protocol = lock_protocol_for_name(optarg);
//			if (protocol < 0)
//				usage("Unknown locking protocol specified.");
//			break;
//		case 'L':
//			cs_length = atof(optarg);
//			if (cs_length <= 0)
//				usage("Invalid critical section length.");
//			break;
//		case 'Q':
//			resource_id = atoi(optarg);
//			if (resource_id <= 0 && strcmp(optarg, "0"))
//				usage("Invalid resource ID.");
//			break;
		case ':':
			usage("Argument missing.");
			break;
		case '?':
		default:
			usage("Bad argument.");
			break;
		}
	}

#ifdef VANILLA_LINUX
	assert(scheduler != LITMUS);
	assert(!wait);
#endif

	assert(stdpct >= 0.0);

	if (MIGRATE_VIA_SYSMEM)
		assert(GPU_PARTITION_SIZE != 1);

	// turn off some features to be safe
	if (scheduler != LITMUS)
	{
		RHO = 0;
		USE_ENGINE_LOCKS = false;
		USE_DYNAMIC_GROUP_LOCKS = false;
		RELAX_FIFO_MAX_LEN = false;
		ENABLE_RT_AUX_THREADS = false;
		budget_ms = -1.0;
		want_enforcement = false;
		want_signals = false;

		cjobfn = job_linux;
		gjobfn = gpu_job_linux;

		if (scheduler == RT_LINUX)
		{
			struct sched_param fifoparams;

			assert(priority >= sched_get_priority_min(SCHED_FIFO) &&
				   priority <= sched_get_priority_max(SCHED_FIFO));

			memset(&fifoparams, 0, sizeof(fifoparams));
			fifoparams.sched_priority = priority;
			assert(0 == sched_setscheduler(getpid(), SCHED_FIFO, &fifoparams));
		}
	}
	else
	{
		cjobfn = job;
		gjobfn = gpu_job;

		if (!litmus_is_valid_fixed_prio(priority))
			usage("Invalid priority.");
	}

	if (test_loop) {
		debug_delay_loop();
		return 0;
	}

	srand(time(0));

	if (file) {
		get_exec_times(file, column, &num_jobs, &exec_times);

		if (argc - optind < 2)
			usage("Arguments missing.");

		for (cur_job = 0; cur_job < num_jobs; ++cur_job) {
			/* convert the execution time to seconds */
			duration += exec_times[cur_job] * 0.001;
		}
	} else {
		/*
		 * if we're not reading from the CSV file, then we need
		 * three parameters
		 */
		if (argc - optind < 3)
			usage("Arguments missing.");
	}

	if (argc - optind == 3) {
		assert(!GPU_USING);
		wcet_ms   = atof(argv[optind + 0]);
		period_ms = atof(argv[optind + 1]);
		duration  = atof(argv[optind + 2]);
	}
	else if (argc - optind == 4) {
		assert(GPU_USING);
		wcet_ms   = atof(argv[optind + 0]);
		gpu_wcet_ms = atof(argv[optind + 1]);
		period_ms = atof(argv[optind + 2]);
		duration  = atof(argv[optind + 3]);
	}

	wcet   = ms2ns(wcet_ms);
	period = ms2ns(period_ms);
	if (wcet <= 0)
		usage("The worst-case execution time must be a "
				"positive number.");
	if (period <= 0)
		usage("The period must be a positive number.");
	if (!file && wcet > period) {
		usage("The worst-case execution time must not "
				"exceed the period.");
	}
	if (GPU_USING && gpu_wcet_ms <= 0)
		usage("The worst-case gpu execution time must be a positive number.");

	if (budget_ms > 0.0)
		budget = ms2ns(budget_ms);
	else
		budget = wcet;

#if 0
	// use upscale to determine breakdown utilization
	// only scaling up CPU time for now.
	double upscale = (double)period/(double)budget - 1.0;
	upscale = std::min(std::max(0.0, upscale), 0.6); // at most 30%
	wcet = wcet + wcet*upscale;
	budget = budget + wcet*upscale;
	wcet_ms = wcet_ms + wcet_ms*upscale;

	// fucking floating point
	if (budget < wcet)
		budget = wcet;
	if (budget > period)
		budget = period;
#endif

	// randomize execution time according to a normal distribution
	// centered around the desired execution time.
	// standard deviation is a percentage of this average
	wcet_dist_ms = new Normal<double>(wcet_ms + gpu_wcet_ms, (wcet_ms + gpu_wcet_ms) * stdpct);
	wcet_dist_ms->seed((unsigned int)time(0));

	if (file && num_jobs > 1)
		duration += period_ms * 0.001 * (num_jobs - 1);

	if (migrate) {
		ret = be_migrate_to_cluster(cluster, cluster_size);
		if (ret < 0)
			bail_out("could not migrate to target partition or cluster.");
	}

	if (scheduler != LITMUS)
	{
		// set some variables needed by linux modes
		if (GPU_USING)
		{
			TRACE_MIGRATIONS = true;
		}
		periodTime.tv_sec = period / s2ns(1);
		periodTime.tv_nsec = period - periodTime.tv_sec * s2ns(1);
		period_ns = period;
	}

	init_rt_task_param(&param);
	param.exec_cost = budget;
	param.period = period;
	param.priority = priority;
	param.cls = cls;
	param.budget_policy = (want_enforcement) ?
			PRECISE_ENFORCEMENT : NO_ENFORCEMENT;
	param.budget_signal_policy = (want_enforcement && want_signals) ?
			PRECISE_SIGNALS : NO_SIGNALS;
	param.drain_policy = drain;
	param.release_policy = PERIODIC;

	if (migrate)
		param.cpu = cluster_to_first_cpu(cluster, cluster_size);
	ret = set_rt_task_param(gettid(), &param);
	if (ret < 0)
		bail_out("could not setup rt task params");

	if (scheduler == LITMUS) {
		init_litmus();
	}
	else {
		init_linux();
	}

	if (want_signals) {
		/* bind default longjmp signal handler to SIG_BUDGET. */
		activate_litmus_signals(SIG_BUDGET_MASK, longjmp_on_litmus_signal);
	}

//	if (protocol >= 0) {
//		/* open reference to semaphore */
//		lock_od = litmus_open_lock(protocol, resource_id, lock_namespace, &cluster);
//		if (lock_od < 0) {
//			perror("litmus_open_lock");
//			usage("Could not open lock.");
//		}
//	}

	if (GPU_USING) {
		allocate_locks(num_gpu_users, scheduler != LITMUS);

		signal(SIGABRT, catch_exit);
		signal(SIGTERM, catch_exit);
		signal(SIGQUIT, catch_exit);
		signal(SIGSEGV, catch_exit);

		init_cuda(num_gpu_users);
		safetynet = true;
	}

	if (scheduler == LITMUS)
	{
		ret = task_mode(LITMUS_RT_TASK);
		if (ret != 0)
			bail_out("could not become RT task");
	}
	else
	{
		trace_name();
		trace_param();
	}

	if (wait) {
		ret = wait_for_ts_release2(&releaseTime);
		if (ret != 0)
			bail_out("wait_for_ts_release2()");

		if (scheduler != LITMUS)
			log_release();
	}
	else if (scheduler != LITMUS)
	{
		clock_gettime(CLOCK_MONOTONIC, &releaseTime);
		sleep_next_period_linux();
	}

	if (scheduler == LITMUS && GPU_USING && ENABLE_RT_AUX_THREADS) {
		if (enable_aux_rt_tasks(AUX_CURRENT | AUX_FUTURE) != 0)
			bail_out("enable_aux_rt_tasks() failed");
	}

	start = wctime();

	if (!GPU_USING) {
		bool keepgoing;
		do
		{
			double job_ms = wcet_dist_ms->random();
			if (job_ms < 0.0)
				job_ms = 0.0;
			keepgoing = cjobfn(ms2s(job_ms * scale), start + duration);
		}while(keepgoing);
	}
	else {
		bool keepgoing;
		do
		{
			double job_ms = wcet_dist_ms->random();
			if (job_ms < 0.0)
				job_ms = 0.0;

			double cpu_job_ms = (job_ms/(wcet_ms + gpu_wcet_ms))*wcet_ms;
			double gpu_job_ms = (job_ms/(wcet_ms + gpu_wcet_ms))*gpu_wcet_ms;
			keepgoing = gjobfn(
							ms2s(cpu_job_ms * scale),
							ms2s(gpu_job_ms * scale),
							num_kernels,
							start + duration);
		}while(keepgoing);
	}

	if (GPU_USING && ENABLE_RT_AUX_THREADS)
		if (disable_aux_rt_tasks(AUX_CURRENT | AUX_FUTURE) != 0)
			bail_out("disable_aux_rt_tasks() failed");

//	if (file) {
//		/* use times read from the CSV file */
//		for (cur_job = 0; cur_job < num_jobs; ++cur_job) {
//			/* convert job's length to seconds */
//			job(exec_times[cur_job] * 0.001 * scale,
//			    start + duration,
//			    lock_od, cs_length * 0.001);
//		}
//	} else {
//		/* convert to seconds and scale */
//	while (job(wcet_ms * 0.001 * scale, start + duration,
//		   lock_od, cs_length * 0.001));
//	}

	if (scheduler == LITMUS)
	{
		ret = task_mode(BACKGROUND_TASK);
		if (ret != 0)
			bail_out("could not become regular task (huh?)");
	}

	if (GPU_USING) {
		safetynet = false;
		exit_cuda();


//		printf("avg: %f\n", ms_sum/gpucount);
	}

	if (wcet_dist_ms)
		delete wcet_dist_ms;

	if (file)
		free(exec_times);

	return 0;
}
