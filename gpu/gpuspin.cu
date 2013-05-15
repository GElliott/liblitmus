#include <sys/time.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <execinfo.h>

#include <exception>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/filesystem.hpp>

#include <random/normal.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "litmus.h"
#include "common.h"

using namespace std;
using namespace boost::interprocess;
using namespace ranlib;

#define ms2s(ms)  ((ms)*0.001)

const unsigned int TOKEN_START = 100;
const unsigned int TOKEN_END = 101;

const unsigned int EE_START = 200;
const unsigned int EE_END = 201;

const unsigned int CE_SEND_START = 300;
const unsigned int CE_SEND_END = 301;

const unsigned int CE_RECV_START = 400;
const unsigned int CE_RECV_END = 401;

bool SILENT = true;
//bool SILENT = false;
inline int xprintf(const char *format, ...)
{
	int ret = 0;
	if (!SILENT) {
		va_list args;
		va_start(args, format);
		ret = vprintf(format, args);
		va_end(args);
	}
	return ret;
}

const char *lock_namespace = "./.gpuspin-locks";
const size_t PAGE_SIZE = sysconf(_SC_PAGESIZE);

const int NR_GPUS = 8;

bool WANT_SIGNALS = false;
inline void gpuspin_block_litmus_signals(unsigned long mask)
{
	if (WANT_SIGNALS)
		block_litmus_signals(mask);
}

inline void gpuspin_unblock_litmus_signals(unsigned long mask)
{
	if (WANT_SIGNALS)
		unblock_litmus_signals(mask);
}

bool GPU_USING = false;
bool ENABLE_AFFINITY = false;
bool RELAX_FIFO_MAX_LEN = false;
bool ENABLE_CHUNKING = false;
bool MIGRATE_VIA_SYSMEM = false;

bool YIELD_LOCKS = false;

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
cudaEvent_t EVENTS[NR_GPUS];
int GPU_HZ[NR_GPUS];
int NUM_SM[NR_GPUS];
int WARP_SIZE[NR_GPUS];
int ELEM_PER_THREAD[NR_GPUS];

enum eScheduler
{
	LITMUS,
	LINUX,
	RT_LINUX
};

struct Args
{
	bool wait;
	bool migrate;
	int cluster;
	int cluster_size;
	bool gpu_using;
	int gpu_partition;
	int gpu_partition_size;
	int rho;
	int num_ce;
	bool reserve_migr_ce;
	bool use_engine_locks;
	eEngineLockTypes engine_lock_type;
	bool yield_locks;
	bool use_dgls;
	eGpuSyncMode gpusync_mode;
	bool enable_affinity;
	int relax_fifo_len;
	eCudaSyncMode sync_mode;
	size_t send_size;
	size_t recv_size;
	size_t state_size;
	bool enable_chunking;
	size_t chunk_size;
	bool use_sysmem_migration;
	int num_kernels;

	double wcet_ms;
	double gpu_wcet_ms;
	double period_ms;

	double budget_ms;

	double stddev;

	eScheduler scheduler;

	unsigned int priority;

	task_class_t cls;

	bool want_enforcement;
	bool want_signals;
	budget_drain_policy_t drain_policy;

	int column;

	int num_gpu_tasks;
	int num_tasks;

	double scale;

	double duration;
};



#define DEFINE_PER_GPU(type, var) type var[NR_GPUS]
#define per_gpu(var, idx) (var[(idx)])
#define this_gpu(var) (var[(CUR_DEVICE)])
#define cur_stream() (this_gpu(STREAMS))
#define cur_event() (this_gpu(EVENTS))
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

		if (num_locks == 1) {
			gpuspin_block_litmus_signals(ALL_LITMUS_SIG_MASKS);
			litmus_lock(locks[0]);
			gpuspin_unblock_litmus_signals(ALL_LITMUS_SIG_MASKS);
		}
		else if(USE_DYNAMIC_GROUP_LOCKS) {
			gpuspin_block_litmus_signals(ALL_LITMUS_SIG_MASKS);
			litmus_dgl_lock(locks, num_locks);
			gpuspin_unblock_litmus_signals(ALL_LITMUS_SIG_MASKS);
		}
		else
		{
			gpuspin_block_litmus_signals(ALL_LITMUS_SIG_MASKS);
			for(int l = 0; l < num_locks; ++l)
			{
				litmus_lock(locks[l]);
			}
			gpuspin_unblock_litmus_signals(ALL_LITMUS_SIG_MASKS);
		}
		locked = true;
	}

	void unlock() {
		if(locks[0] == locks[1]) crash();

		if (num_locks == 1) {
			gpuspin_block_litmus_signals(ALL_LITMUS_SIG_MASKS);
			litmus_unlock(locks[0]);
			gpuspin_unblock_litmus_signals(ALL_LITMUS_SIG_MASKS);
		}
		else if(USE_DYNAMIC_GROUP_LOCKS) {
			gpuspin_block_litmus_signals(ALL_LITMUS_SIG_MASKS);
			litmus_dgl_unlock(locks, num_locks);
			gpuspin_unblock_litmus_signals(ALL_LITMUS_SIG_MASKS);
		}
		else
		{
			gpuspin_block_litmus_signals(ALL_LITMUS_SIG_MASKS);
			// reverse order
			for(int l = num_locks - 1; l >= 0; --l)
			{
				litmus_unlock(locks[l]);
			}
			gpuspin_unblock_litmus_signals(ALL_LITMUS_SIG_MASKS);
		}
		locked = false;
	}

	bool should_yield() {
		int yield = 1; // assume we should yield
		if (YIELD_LOCKS) {
			if(locks[0] == locks[1]) crash();
			if (num_locks == 1)
				yield = litmus_should_yield_lock(locks[0]);
			else if(USE_DYNAMIC_GROUP_LOCKS)
				yield = litmus_dgl_should_yield_lock(locks, num_locks);
			else
				for(int l = num_locks - 1; l >= 0; --l)  // reverse order
					yield = litmus_should_yield_lock(locks[l]);
		}
		return (yield);
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

		if (state && state->locked) {
			// we have to unlock/re-lock the copy engine to refresh our budget unless
			// we still have budget available.
			if (!state->budgetIsAvailable(bytesToCopy)) {
				// optimization - don't unlock if no one else needs the engine
				if (state->should_yield()) {
					gpuspin_block_litmus_signals(ALL_LITMUS_SIG_MASKS); 
					cudaEventSynchronize(EVENTS[CUR_DEVICE]);
					ret = cudaGetLastError();
					if (kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice)
						inject_action(CE_RECV_END);
					if (kind == cudaMemcpyHostToDevice)
						inject_action(CE_SEND_END);
					gpuspin_unblock_litmus_signals(ALL_LITMUS_SIG_MASKS);

					state->unlock();
					if(ret != cudaSuccess)
						break;
				}
				// we can only run out of
				// budget if chunking is enabled.
				// we presume that init budget would
				// be set to cover entire memcpy
				// if chunking were disabled.
				state->refresh();
			}
		}

		if(state && !state->locked) {
			state->lock();
			if (kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice)
				inject_action(CE_RECV_START);
			if (kind == cudaMemcpyHostToDevice)
				inject_action(CE_SEND_START);
		}

        //ret = cudaMemcpy(dst+i*chunk_size, src+i*chunk_size, bytesToCopy, kind);
		gpuspin_block_litmus_signals(ALL_LITMUS_SIG_MASKS);
		cudaMemcpyAsync(dst+i*chunk_size, src+i*chunk_size, bytesToCopy, kind, STREAMS[CUR_DEVICE]);
		cudaEventRecord(EVENTS[CUR_DEVICE], STREAMS[CUR_DEVICE]);
		gpuspin_unblock_litmus_signals(ALL_LITMUS_SIG_MASKS);

		if(state)
			state->decreaseBudget(bytesToCopy);

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
		gpuspin_block_litmus_signals(ALL_LITMUS_SIG_MASKS);
		cudaEventSynchronize(cur_event());
		if(ret == cudaSuccess)
			ret = cudaGetLastError();
		gpuspin_unblock_litmus_signals(ALL_LITMUS_SIG_MASKS);
	}
	else {
		ce_lock_state state(device_a, kind, count, device_b, migration);
		state.lock();

		if (kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice)
			inject_action(CE_RECV_START);
		if (kind == cudaMemcpyHostToDevice)
			inject_action(CE_SEND_START);

		ret = __chunkMemcpy(a_dst, a_src, count, kind, &state);
		gpuspin_block_litmus_signals(ALL_LITMUS_SIG_MASKS);
		cudaEventSynchronize(cur_event());
		//		cudaStreamSynchronize(cur_stream());
		if(ret == cudaSuccess)
			ret = cudaGetLastError();

		if (kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyDeviceToDevice)
			inject_action(CE_RECV_END);
		if (kind == cudaMemcpyHostToDevice)
			inject_action(CE_SEND_END);
		gpuspin_unblock_litmus_signals(ALL_LITMUS_SIG_MASKS);

		state.unlock();
	}
	return ret;
}

int LITMUS_LOCK_FD = 0;

int EXP_OFFSET = 0;

void allocate_locks_litmus(void)
{
	stringstream ss;
	ss<<lock_namespace<<"-"<<EXP_OFFSET;

	// allocate k-FMLP lock
	//LITMUS_LOCK_FD = open(lock_namespace, O_RDONLY | O_CREAT, S_IRUSR | S_IWUSR);
	LITMUS_LOCK_FD = open(ss.str().c_str(), O_RDONLY | O_CREAT, S_IRUSR | S_IWUSR);
	int *fd = &LITMUS_LOCK_FD;

	int base_name = GPU_PARTITION * 100 + EXP_OFFSET * 200;
	++EXP_OFFSET;

	if (GPU_SYNC_MODE == IKGLP_MODE) {
		/* Standard (optimal) IKGLP */
		TOKEN_LOCK = open_gpusync_token_lock(*fd,
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
		TOKEN_LOCK = open_gpusync_token_lock(*fd,
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
		TOKEN_LOCK = open_gpusync_token_lock(*fd,
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
		TOKEN_LOCK = open_gpusync_token_lock(*fd,
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

			ee_lock = openEngineLock(*fd, ee_name);
			if (ee_lock < 0)
				perror("open_*_sem (engine lock)");

			ce_0_lock = openEngineLock(*fd, ce_0_name);
			if (ce_0_lock < 0)
				perror("open_*_sem (engine lock)");

			if (NUM_COPY_ENGINES == 2)
			{
				ce_1_lock = openEngineLock(*fd, ce_1_name);
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

void deallocate_locks_litmus(void)
{
	for (int i = 0; i < GPU_PARTITION_SIZE; ++i)
	{
		int idx = GPU_PARTITION*GPU_PARTITION_SIZE + i;

		od_close(EE_LOCKS[idx]);
		if (NUM_COPY_ENGINES == 1)
		{
			od_close(CE_SEND_LOCKS[idx]);
		}
		else
		{
			if (RESERVED_MIGR_COPY_ENGINE) {
				od_close(CE_SEND_LOCKS[idx]);
				od_close(CE_MIGR_SEND_LOCKS[idx]);
			}
			else {
				od_close(CE_SEND_LOCKS[idx]);
				od_close(CE_RECV_LOCKS[idx]);
			}
		}
	}

	od_close(TOKEN_LOCK);

	close(LITMUS_LOCK_FD);

	memset(&CE_SEND_LOCKS[0], 0, sizeof(CE_SEND_LOCKS));
	memset(&CE_RECV_LOCKS[0], 0, sizeof(CE_RECV_LOCKS));
	memset(&CE_MIGR_SEND_LOCKS[0], 0, sizeof(CE_MIGR_SEND_LOCKS));
	memset(&CE_MIGR_RECV_LOCKS[0], 0, sizeof(CE_MIGR_RECV_LOCKS));
	TOKEN_LOCK = -1;
	LITMUS_LOCK_FD = 0;
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
		int last = (ENABLE_AFFINITY) ?
				((preference >= 0) ? preference : 0) :
				(rand()%poolSize);
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


static managed_shared_memory *linux_lock_segment_ptr = NULL;
static gpu_pool* GPU_LINUX_SEM_POOL = NULL;
static pthread_mutex_t* GPU_LINUX_MUTEX_POOL = NULL;

static void allocate_locks_linux(const int num_gpu_users)
{
	int numGpuPartitions = NR_GPUS/GPU_PARTITION_SIZE;

	if(num_gpu_users > 0)
	{
		xprintf("%d: creating linux locks\n", getpid());
		shared_memory_object::remove("linux_lock_memory");

		linux_lock_segment_ptr = new managed_shared_memory(create_only, "linux_lock_memory", 30*PAGE_SIZE);
		GPU_LINUX_MUTEX_POOL = linux_lock_segment_ptr->construct<pthread_mutex_t>("pthread_mutex_t linux_m")[numGpuPartitions]();
		for(int i = 0; i < numGpuPartitions; ++i)
		{
			pthread_mutexattr_t attr;
			pthread_mutexattr_init(&attr);
			pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
			pthread_mutex_init(&(GPU_LINUX_MUTEX_POOL[i]), &attr);
			pthread_mutexattr_destroy(&attr);
		}
		GPU_LINUX_SEM_POOL = linux_lock_segment_ptr->construct<gpu_pool>("gpu_pool linux_p")[numGpuPartitions](GPU_PARTITION_SIZE);
	}
	else
	{
		sleep(5);
		do
		{
			try
			{
				if (!linux_lock_segment_ptr)
					linux_lock_segment_ptr = new managed_shared_memory(open_only, "linux_lock_memory");
			}
			catch(...)
			{
				sleep(1);
			}
		}while(linux_lock_segment_ptr == NULL);

		GPU_LINUX_MUTEX_POOL = linux_lock_segment_ptr->find<pthread_mutex_t>("pthread_mutex_t linux_m").first;
		GPU_LINUX_SEM_POOL = linux_lock_segment_ptr->find<gpu_pool>("gpu_pool linux_p").first;
	}
}

static void deallocate_locks_linux(const int num_gpu_users)
{
	GPU_LINUX_MUTEX_POOL = NULL;
	GPU_LINUX_SEM_POOL = NULL;

	delete linux_lock_segment_ptr;
	linux_lock_segment_ptr = NULL;

	if(num_gpu_users > 0)
		shared_memory_object::remove("linux_lock_memory");
}




static void allocate_locks(const int num_gpu_users, bool linux_mode)
{
	if(!linux_mode)
		allocate_locks_litmus();
	else
		allocate_locks_linux(num_gpu_users);
}

static void deallocate_locks(const int num_gpu_users, bool linux_mode)
{
	if(!linux_mode)
		deallocate_locks_litmus();
	else
		deallocate_locks_linux(num_gpu_users);
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


//static pthread_barrier_t *gpu_barrier = NULL;
static interprocess_mutex *gpu_mgmt_mutexes = NULL;
static managed_shared_memory *gpu_mutex_segment_ptr = NULL;

void coordinate_gpu_tasks(const int num_gpu_users)
{
	if(num_gpu_users > 0)
	{
		xprintf("%d creating shared memory\n", getpid());
		shared_memory_object::remove("gpu_mutex_memory");
		gpu_mutex_segment_ptr = new managed_shared_memory(create_only, "gpu_mutex_memory", PAGE_SIZE);

//		printf("%d creating a barrier for %d users\n", getpid(), num_gpu_users);
//		gpu_barrier = segment_ptr->construct<pthread_barrier_t>("pthread_barrier_t gpu_barrier")();
//		pthread_barrierattr_t battr;
//		pthread_barrierattr_init(&battr);
//		pthread_barrierattr_setpshared(&battr, PTHREAD_PROCESS_SHARED);
//		pthread_barrier_init(gpu_barrier, &battr, num_gpu_users);
//		pthread_barrierattr_destroy(&battr);
//		printf("%d creating gpu mgmt mutexes for %d devices\n", getpid(), NR_GPUS);
		gpu_mgmt_mutexes = gpu_mutex_segment_ptr->construct<interprocess_mutex>("interprocess_mutex m")[NR_GPUS]();
	}
	else
	{
		sleep(5);
		do
		{
			try
			{
				gpu_mutex_segment_ptr = new managed_shared_memory(open_only, "gpu_mutex_memory");
			}
			catch(...)
			{
				sleep(1);
			}
		}while(gpu_mutex_segment_ptr == NULL);

//		gpu_barrier = segment_ptr->find<pthread_barrier_t>("pthread_barrier_t gpu_barrier").first;
		gpu_mgmt_mutexes = gpu_mutex_segment_ptr->find<interprocess_mutex>("interprocess_mutex m").first;
	}
}

const size_t SEND_ALLOC_SIZE = 12*1024;
const size_t RECV_ALLOC_SIZE = 12*1024;
const size_t STATE_ALLOC_SIZE = 16*1024;

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

static void destroy_events()
{
	for(int i = 0; i < GPU_PARTITION_SIZE; ++i)
	{
		int which = GPU_PARTITION*GPU_PARTITION_SIZE + i;
		gpu_mgmt_mutexes[which].lock();
		set_cur_gpu(which);
		cudaEventDestroy(EVENTS[which]);
		gpu_mgmt_mutexes[which].unlock();
	}
}

static void init_events()
{
	xprintf("creating %s events\n", (CUDA_SYNC_MODE == BLOCKING) ? "blocking" : "spinning");
	for(int i = 0; i < GPU_PARTITION_SIZE; ++i)
	{
		int which = GPU_PARTITION*GPU_PARTITION_SIZE + i;
		gpu_mgmt_mutexes[which].lock();
		set_cur_gpu(which);
		if (CUDA_SYNC_MODE == BLOCKING)
			cudaEventCreateWithFlags(&EVENTS[which], cudaEventBlockingSync | cudaEventDisableTiming);
		else
			cudaEventCreateWithFlags(&EVENTS[which], cudaEventDefault | cudaEventDisableTiming);
		gpu_mgmt_mutexes[which].unlock();
	}
}

static void init_cuda(const int num_gpu_users)
{
	size_t send_alloc_bytes = SEND_ALLOC_SIZE + (SEND_ALLOC_SIZE%PAGE_SIZE != 0)*PAGE_SIZE;
	size_t recv_alloc_bytes = RECV_ALLOC_SIZE + (RECV_ALLOC_SIZE%PAGE_SIZE != 0)*PAGE_SIZE;
	size_t state_alloc_bytes = STATE_ALLOC_SIZE + (STATE_ALLOC_SIZE%PAGE_SIZE != 0)*PAGE_SIZE;

	static bool first_time = true;

	if (first_time) {
		coordinate_gpu_tasks(num_gpu_users);
		first_time = false;
	}

#if 0
	switch (CUDA_SYNC_MODE)
	{
		case BLOCKING:
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			break;
		case SPIN:
			cudaSetDeviceFlags(cudaDeviceScheduleSpin);
			break;
	}
#endif

	for(int i = 0; i < GPU_PARTITION_SIZE; ++i)
	{
		cudaDeviceProp prop;
		int which = GPU_PARTITION*GPU_PARTITION_SIZE + i;

		gpu_mgmt_mutexes[which].lock();
		try
		{
			set_cur_gpu(which);

			xprintf("setting up GPU %d\n", which);

			cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 0);
			cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0);

			cudaGetDeviceProperties(&prop, which);
			GPU_HZ[which] = prop.clockRate * 1000; /* khz -> hz */
			NUM_SM[which] = prop.multiProcessorCount;
			WARP_SIZE[which] = prop.warpSize;

			// enough to fill the L2 cache exactly.
			ELEM_PER_THREAD[which] = (prop.l2CacheSize/(NUM_SM[which]*WARP_SIZE[which]*sizeof(spindata_t)));

//			if (!MIGRATE_VIA_SYSMEM && prop.unifiedAddressing)
			if (prop.unifiedAddressing)
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

			cudaStreamCreate(&STREAMS[which]);

			// gpu working set
			cudaMalloc(&d_spin_data[which], prop.l2CacheSize);
			cudaMemset(&d_spin_data[which], 0, prop.l2CacheSize);

			// send data
			cudaMalloc(&d_send_data[which], send_alloc_bytes);
			cudaHostAlloc(&h_send_data, send_alloc_bytes, cudaHostAllocPortable | cudaHostAllocMapped);

			// recv data
			cudaMalloc(&d_recv_data[which], recv_alloc_bytes);
			cudaHostAlloc(&h_recv_data, recv_alloc_bytes, cudaHostAllocPortable | cudaHostAllocMapped);

			// state data
			cudaMalloc(&d_state_data[which], state_alloc_bytes);
			cudaHostAlloc(&h_state_data, state_alloc_bytes, cudaHostAllocPortable | cudaHostAllocMapped | cudaHostAllocWriteCombined);
		}
		catch(std::exception &e)
		{
			fprintf(stderr, "caught an exception during initializiation!: %s\n", e.what());
		}
		catch(...)
		{
			fprintf(stderr, "caught unknown exception.\n");
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
	fprintf(stderr, "Tried to sysmem migrate from %d to %d\n",
					from, to);
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
	else if(cur_gpu() == -1) {
		set_cur_gpu(next_gpu);
	}
}

static void exit_cuda()
{
#if 0
	for(int i = 0; i < GPU_PARTITION_SIZE; ++i)
	{
		int which = GPU_PARTITION*GPU_PARTITION_SIZE + i;
		gpu_mgmt_mutexes[which].lock();
		set_cur_gpu(which);
		cudaFree(d_send_data[which]);
		cudaFree(d_recv_data[which]);
		cudaFree(d_state_data[which]);
		cudaFree(d_spin_data[which]);
		gpu_mgmt_mutexes[which].unlock();
	}
#endif

	cudaFreeHost(h_send_data);
	cudaFreeHost(h_recv_data);
	cudaFreeHost(h_state_data);

	for(int i = 0; i < GPU_PARTITION_SIZE; ++i)
	{
		int which = GPU_PARTITION*GPU_PARTITION_SIZE + i;
		gpu_mgmt_mutexes[which].lock();
		set_cur_gpu(which);
		cudaDeviceReset();
		gpu_mgmt_mutexes[which].unlock();
	}

	memset(d_send_data, 0, sizeof(d_send_data));
	memset(d_recv_data, 0, sizeof(d_recv_data));
	memset(d_state_data, 0, sizeof(d_state_data));
	memset(d_spin_data, 0, sizeof(d_spin_data));
	h_send_data = NULL;
	h_recv_data = NULL;
	h_state_data = NULL;
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
	bool ee_locked = false;
	bool early_exit = false;

	if (gpu_sec_time <= 0.0)
		goto out;
	if (emergency_exit && wctime() > emergency_exit)
		goto out;

	gpuspin_block_litmus_signals(ALL_LITMUS_SIG_MASKS);
	next_gpu = litmus_lock(TOKEN_LOCK);
	inject_action(TOKEN_START);
	gpuspin_unblock_litmus_signals(ALL_LITMUS_SIG_MASKS);

	LITMUS_TRY
	{
		gpuspin_block_litmus_signals(ALL_LITMUS_SIG_MASKS);
		MigrateIfNeeded(next_gpu);
		gpuspin_unblock_litmus_signals(ALL_LITMUS_SIG_MASKS);

		unsigned int numcycles = ((unsigned int)(cur_hz() * gpu_sec_time))/num_kernels;

		if(SEND_SIZE > 0)
			chunkMemcpy(this_gpu(d_state_data), h_send_data, SEND_SIZE,
						cudaMemcpyHostToDevice, CUR_DEVICE, useEngineLocks());

		for(unsigned int i = 0; i < num_kernels; ++i)
		{
			gpuspin_block_litmus_signals(ALL_LITMUS_SIG_MASKS);

			if(useEngineLocks() && !ee_locked) {
				litmus_lock(cur_ee());
				inject_action(EE_START);
				ee_locked = true;
			}
			/* one block per sm, one warp per block */
			docudaspin <<<cur_sms(), cur_warp_size(), 0, cur_stream()>>> (d_spin_data[cur_gpu()], cur_elem_per_thread(), numcycles);
			if(useEngineLocks() && (!YIELD_LOCKS || (YIELD_LOCKS && litmus_should_yield_lock(cur_ee())))) {
//				cudaStreamSynchronize(cur_stream());
				cudaEventRecord(cur_event(), cur_stream());
				cudaEventSynchronize(cur_event());
				inject_action(EE_END);
				litmus_unlock(cur_ee());
				ee_locked = false;
			}
			gpuspin_unblock_litmus_signals(ALL_LITMUS_SIG_MASKS);
		}

		if (ee_locked) {
			gpuspin_block_litmus_signals(ALL_LITMUS_SIG_MASKS);

			cudaEventRecord(cur_event(), cur_stream());
			cudaEventSynchronize(cur_event());
			inject_action(EE_END);
			litmus_unlock(cur_ee());

			gpuspin_unblock_litmus_signals(ALL_LITMUS_SIG_MASKS);
			ee_locked = false;
		}

		if(RECV_SIZE > 0)
			chunkMemcpy(h_recv_data, this_gpu(d_state_data), RECV_SIZE,
						cudaMemcpyDeviceToHost, CUR_DEVICE, useEngineLocks());

		if (MIGRATE_VIA_SYSMEM) {
			gpuspin_block_litmus_signals(ALL_LITMUS_SIG_MASKS);
			PullState();
			gpuspin_unblock_litmus_signals(ALL_LITMUS_SIG_MASKS);
		}
	}
	LITMUS_CATCH(SIG_BUDGET)
	{
			cudaEventRecord(cur_event(), cur_stream());
			cudaEventSynchronize(cur_event());

			if (useEngineLocks()) {
				/* unlock all engine locks. will fail safely if not held */
				litmus_unlock(cur_ee());
				if (NUM_COPY_ENGINES == 1) {
					litmus_unlock(cur_send());
				}
				else if (RESERVED_MIGR_COPY_ENGINE) {
					litmus_unlock(cur_send());
					litmus_unlock(cur_migr_send());
				}
				else {
					litmus_unlock(cur_send());
					litmus_unlock(cur_recv());
				}
			}
			early_exit = true;
	}
	END_LITMUS_TRY

	gpuspin_block_litmus_signals(ALL_LITMUS_SIG_MASKS);
	inject_action(TOKEN_END);
	litmus_unlock(TOKEN_LOCK);
	gpuspin_unblock_litmus_signals(ALL_LITMUS_SIG_MASKS);

	last_gpu() = cur_gpu();

	if (early_exit)
		throw std::exception();

out:
	return;
}

static void gpu_loop_for_linux(double gpu_sec_time, unsigned int num_kernels, double emergency_exit)
{
	int GPU_OFFSET = GPU_PARTITION * GPU_PARTITION_SIZE;
	gpu_pool *pool = &GPU_LINUX_SEM_POOL[GPU_PARTITION];
	pthread_mutex_t *mutex = &GPU_LINUX_MUTEX_POOL[GPU_PARTITION];

	int next_gpu;

	if (gpu_sec_time <= 0.0)
		goto out;
	if (emergency_exit && wctime() > emergency_exit)
		goto out;

	next_gpu = pool->get(mutex, ((cur_gpu() != -1) ?
					 	cur_gpu() - GPU_OFFSET :
						-1))
				+ GPU_OFFSET;
	{
		MigrateIfNeeded(next_gpu);

		unsigned int numcycles = ((unsigned int)(cur_hz() * gpu_sec_time))/num_kernels;

		if(SEND_SIZE > 0)
			chunkMemcpy(this_gpu(d_state_data), h_send_data, SEND_SIZE,
						cudaMemcpyHostToDevice, cur_gpu(), useEngineLocks());

		for(unsigned int i = 0; i < num_kernels; ++i)
		{
			/* one block per sm, one warp per block */
			docudaspin <<<cur_sms(),cur_warp_size(), 0, cur_stream()>>> (d_spin_data[cur_gpu()], cur_elem_per_thread(), numcycles);
			cudaEventRecord(cur_event(), cur_stream());
			cudaEventSynchronize(cur_event());
//			cudaStreamSynchronize(cur_stream());
		}

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

///*
// * returns the character that made processing stop, newline or EOF
// */
//static int skip_to_next_line(FILE *fstream)
//{
//	int ch;
//	for (ch = fgetc(fstream); ch != EOF && ch != '\n'; ch = fgetc(fstream));
//	return ch;
//}
//
//static void skip_comments(FILE *fstream)
//{
//	int ch;
//	for (ch = fgetc(fstream); ch == '#'; ch = fgetc(fstream))
//		skip_to_next_line(fstream);
//	ungetc(ch, fstream);
//}
//
//static void get_exec_times(const char *file, const int column,
//			   int *num_jobs,    double **exec_times)
//{
//	FILE *fstream;
//	int  cur_job, cur_col, ch;
//	*num_jobs = 0;
//
//	fstream = fopen(file, "r");
//	if (!fstream)
//		bail_out("could not open execution time file");
//
//	/* figure out the number of jobs */
//	do {
//		skip_comments(fstream);
//		ch = skip_to_next_line(fstream);
//		if (ch != EOF)
//			++(*num_jobs);
//	} while (ch != EOF);
//
//	if (-1 == fseek(fstream, 0L, SEEK_SET))
//		bail_out("rewinding file failed");
//
//	/* allocate space for exec times */
//	*exec_times = (double*)calloc(*num_jobs, sizeof(*exec_times));
//	if (!*exec_times)
//		bail_out("couldn't allocate memory");
//
//	for (cur_job = 0; cur_job < *num_jobs && !feof(fstream); ++cur_job) {
//
//		skip_comments(fstream);
//
//		for (cur_col = 1; cur_col < column; ++cur_col) {
//			/* discard input until we get to the column we want */
//			int unused __attribute__ ((unused)) = fscanf(fstream, "%*s,");
//		}
//
//		/* get the desired exec. time */
//		if (1 != fscanf(fstream, "%lf", (*exec_times)+cur_job)) {
//			fprintf(stderr, "invalid execution time near line %d\n",
//					cur_job);
//			exit(EXIT_FAILURE);
//		}
//
//		skip_to_next_line(fstream);
//	}
//
//	assert(cur_job == *num_jobs);
//	fclose(fstream);
//}

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


//static void debug_delay_loop(void)
//{
//	double start, end, delay;
//
//	while (1) {
//		for (delay = 0.5; delay > 0.01; delay -= 0.01) {
//			start = wctime();
//			loop_for(delay, 0);
//			end = wctime();
//			printf("%6.4fs: looped for %10.8fs, delta=%11.8fs, error=%7.4f%%\n",
//			       delay,
//			       end - start,
//			       end - start - delay,
//			       100 * (end - start - delay) / delay);
//		}
//	}
//}

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

		LITMUS_TRY
		{
			try
			{
				loop_for(chunk1, program_end + 1);
				gpu_loop_for(gpu_exec_time, num_kernels, program_end + 1);
				loop_for(chunk2, program_end + 1);
			}
			catch(std::exception& e)
			{
				xprintf("%d: ran out of time while using GPU\n", gettid());
			}
		}
		LITMUS_CATCH(SIG_BUDGET)
		{
			xprintf("%d: ran out of time\n", gettid());
		}
		END_LITMUS_TRY

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
		LITMUS_TRY
		{
			loop_for(exec_time, program_end + 1);
		}
		LITMUS_CATCH(SIG_BUDGET)
		{
			xprintf("%d: ran out of time\n", gettid());
		}
		END_LITMUS_TRY
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

static int enable_aux_rt_tasks_linux(pid_t tid)
{
	/* pre: caller must already be real time */
	int ret = 0;
	struct sched_param param;
	stringstream pidstr;
	boost::filesystem::directory_iterator theEnd;
	boost::filesystem::path proc_dir;

	int policy = sched_getscheduler(tid);
	if (policy == -1 || policy != SCHED_FIFO) {
		ret = -1;
		goto out;
	}

	ret = sched_getparam(tid, &param);
	if (ret < 0)
		goto out;


	pidstr<<getpid();
	proc_dir = boost::filesystem::path("/proc");
	proc_dir /= pidstr.str();
	proc_dir /= "task";

	for(boost::filesystem::directory_iterator iter(proc_dir); iter != theEnd; ++iter)
	{
		stringstream taskstr(iter->path().leaf().c_str());
		int child = 0;
		taskstr>>child;
		if (child != tid && child != 0)
		{
			/* mirror tid's params to others */
			ret = sched_setscheduler(child, policy, &param);
			if (ret != 0)
				goto out;
		}
	}

out:
	return ret;
}

static int disable_aux_rt_tasks_linux(pid_t tid)
{
	int ret = 0;
	struct sched_param param;
	stringstream pidstr;
	boost::filesystem::directory_iterator theEnd;
	boost::filesystem::path proc_dir;

	memset(&param, 0, sizeof(param));

	pidstr<<getpid();
	proc_dir = boost::filesystem::path("/proc");
	proc_dir /= pidstr.str();
	proc_dir /= "task";

	for(boost::filesystem::directory_iterator iter(proc_dir); iter != theEnd; ++iter)
	{
		stringstream taskstr(iter->path().leaf().c_str());
		int child = 0;
		taskstr>>child;
		if (child != tid && child != 0)
		{
			/* make all other threads sched_normal */
			ret = sched_setscheduler(child, SCHED_OTHER, &param);
			if (ret != 0)
				goto out;
		}
	}

out:
	return ret;
}

static int be_migrate_all_to_cluster(int cluster, int cluster_size)
{
	int ret = 0;
	stringstream pidstr;

	pidstr<<getpid();
	boost::filesystem::path proc_dir("/proc");
	proc_dir /= pidstr.str();
	proc_dir /= "task";
	boost::filesystem::directory_iterator theEnd;
	for(boost::filesystem::directory_iterator iter(proc_dir); iter != theEnd; ++iter)
	{
		stringstream taskstr(iter->path().leaf().c_str());
		int task = 0;
		taskstr>>task;
		if (task != 0) {
			ret = be_migrate_to_cluster(cluster, cluster_size);
			if (ret != 0)
				goto out;
		}
	}

out:
	return ret;
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





enum eRunMode
{
	NORMAL,
	PROXY,
	DAEMON,
};

void set_defaults(struct Args* args)
{
	memset(args, 0, sizeof(*args));
	args->wcet_ms = -1.0;
	args->gpu_wcet_ms = 0.0;
	args->period_ms = -1.0;
	args->budget_ms = -1.0;
	args->gpusync_mode = IKGLP_MODE;
	args->sync_mode = BLOCKING;
	args->gpu_using = false;
	args->enable_affinity = false;
	args->enable_chunking = false;
	args->relax_fifo_len = false;
	args->use_sysmem_migration = false;
	args->rho = 2;
	args->num_ce = 2;
	args->reserve_migr_ce = false;
	args->num_kernels = 1;
	args->engine_lock_type = FIFO;
	args->yield_locks = false;
	args->drain_policy = DRAIN_SIMPLE;
	args->want_enforcement = false;
	args->want_signals = false;
	args->priority = LITMUS_LOWEST_PRIORITY;
	args->cls = RT_CLASS_SOFT;
	args->scheduler = LITMUS;
	args->migrate = false;
	args->cluster = 0;
	args->cluster_size = 1;
	args->stddev = 0.0;
	args->wait = false;
	args->scale = 1.0;
	args->duration = 0.0;
}

void apply_args(struct Args* args)
{
	// set all the globals
	CPU_PARTITION_SIZE = args->cluster_size;
	GPU_USING = args->gpu_using;
	GPU_PARTITION = args->gpu_partition;
	GPU_PARTITION_SIZE = args->gpu_partition_size;
	RHO = args->rho;
	NUM_COPY_ENGINES = args->num_ce;
	RESERVED_MIGR_COPY_ENGINE = args->reserve_migr_ce;
	USE_ENGINE_LOCKS = args->use_engine_locks;
	ENGINE_LOCK_TYPE = args->engine_lock_type;
	YIELD_LOCKS = args->yield_locks;
	USE_DYNAMIC_GROUP_LOCKS = args->use_dgls;
	GPU_SYNC_MODE = args->gpusync_mode;
	ENABLE_AFFINITY = args->enable_affinity;
	RELAX_FIFO_MAX_LEN = args->relax_fifo_len;
	CUDA_SYNC_MODE = args->sync_mode;
	SEND_SIZE = args->send_size;
	RECV_SIZE = args->recv_size;
	STATE_SIZE = args->state_size;
	ENABLE_CHUNKING = args->enable_chunking;
	CHUNK_SIZE = args->chunk_size;
	MIGRATE_VIA_SYSMEM = args->use_sysmem_migration;

	if (args->scheduler == LITMUS && !ENABLE_AFFINITY)
		TRACE_MIGRATIONS = true;
	else if (args->scheduler == LITMUS)
		TRACE_MIGRATIONS = false;

	WANT_SIGNALS = args->want_signals;

	// roll back other globals to an initial state
	CUR_DEVICE = -1;
	LAST_DEVICE = -1;
}

int __do_normal(struct Args* args)
{
	int ret = 0;
	struct rt_task param;

	lt_t wcet;
	lt_t period;
	lt_t budget;

	Normal<double> *wcet_dist_ms = NULL;

	cpu_job_t cjobfn = NULL;
	gpu_job_t gjobfn = NULL;

	double start = 0;

	if (MIGRATE_VIA_SYSMEM && GPU_PARTITION_SIZE == 1)
		return -1;

	// turn off some features to be safe
	if (args->scheduler != LITMUS)
	{
		RHO = 0;
		USE_ENGINE_LOCKS = false;
		USE_DYNAMIC_GROUP_LOCKS = false;
		RELAX_FIFO_MAX_LEN = false;
		ENABLE_RT_AUX_THREADS = false;
		args->want_enforcement = false;
		args->want_signals = false;

		cjobfn = job_linux;
		gjobfn = gpu_job_linux;
	}
	else
	{
		cjobfn = job;
		gjobfn = gpu_job;
	}

	wcet   = ms2ns(args->wcet_ms);
	period = ms2ns(args->period_ms);

	if (wcet <= 0) {
		fprintf(stderr, "The worst-case execution time must be a positive number.\n");
		ret = -1;
		goto out;
	}
	if (period <= 0) {
		fprintf(stderr, "The period must be a positive number.\n");
		ret = -1;
		goto out;
	}
	if (wcet > period) {
		fprintf(stderr, "The worst-case execution time must not exceed the period.\n");
		ret = -1;
		goto out;
	}
	if (args->gpu_using && args->gpu_wcet_ms <= 0) {
		fprintf(stderr, "The worst-case gpu execution time must be a positive number.\n");
		ret = -1;
		goto out;
	}

	if (args->budget_ms > 0.0)
		budget = ms2ns(args->budget_ms);
	else
		budget = wcet;

	// randomize execution time according to a normal distribution
	// centered around the desired execution time.
	// standard deviation is a percentage of this average
	wcet_dist_ms = new Normal<double>(args->wcet_ms + args->gpu_wcet_ms, (args->wcet_ms + args->gpu_wcet_ms) * args->stddev);
	wcet_dist_ms->seed((unsigned int)time(0));

	ret = be_migrate_all_to_cluster(args->cluster, args->cluster_size);
	if (ret < 0) {
		fprintf(stderr, "could not migrate to target partition or cluster.\n");
		goto out;
	}

	if (args->scheduler != LITMUS)
	{
		// set some variables needed by linux modes
		if (args->gpu_using)
			TRACE_MIGRATIONS = true;
		periodTime.tv_sec = period / s2ns(1);
		periodTime.tv_nsec = period - periodTime.tv_sec * s2ns(1);
		period_ns = period;
		job_no = 0;
	}

	init_rt_task_param(&param);
	param.exec_cost = budget;
	param.period = period;
	param.priority = args->priority;
	param.cls = args->cls;
	param.budget_policy = (args->want_enforcement) ?
		PRECISE_ENFORCEMENT : NO_ENFORCEMENT;
	param.budget_signal_policy = (args->want_signals) ?
		PRECISE_SIGNALS : NO_SIGNALS;
	param.drain_policy = args->drain_policy;
	param.drain_policy = args->drain_policy;
	param.release_policy = PERIODIC;
	param.cpu = cluster_to_first_cpu(args->cluster, args->cluster_size);

	ret = set_rt_task_param(gettid(), &param);
	if (ret < 0) {
		bail_out("could not setup rt task params\n");
		goto out;
	}

	if (args->want_signals)
		/* bind default longjmp signal handler to SIG_BUDGET. */
		activate_litmus_signals(SIG_BUDGET_MASK, longjmp_on_litmus_signal);
	else
		ignore_litmus_signals(SIG_BUDGET_MASK);

	if (args->gpu_using)
		allocate_locks(args->num_gpu_tasks, args->scheduler != LITMUS);

	if (args->scheduler == LITMUS)
	{
		ret = task_mode(LITMUS_RT_TASK);
		if (ret < 0) {
			fprintf(stderr, "could not become RT task\n");
			goto out;
		}
	}
	else
	{
		if (args->scheduler == RT_LINUX)
		{
			struct sched_param fifoparams;
			memset(&fifoparams, 0, sizeof(fifoparams));
			fifoparams.sched_priority = args->priority;
			ret = sched_setscheduler(getpid(), SCHED_FIFO, &fifoparams);
			if (ret < 0) {
				fprintf(stderr, "could not become sched_fifo task\n");
				goto out;
			}
		}
		trace_name();
		trace_param();
	}

	if (args->wait) {
		xprintf("%d: waiting for release.\n", getpid());
		ret = wait_for_ts_release2(&releaseTime);
		if (ret != 0) {
			printf("wait_for_ts_release2()\n");
			goto out;
		}

		if (args->scheduler != LITMUS)
			log_release();
	}
	else if (args->scheduler != LITMUS)
	{
		clock_gettime(CLOCK_MONOTONIC, &releaseTime);
		sleep_next_period_linux();
	}

	if (args->gpu_using && ENABLE_RT_AUX_THREADS) {
		if (args->scheduler == LITMUS) {
			ret = enable_aux_rt_tasks(AUX_CURRENT | AUX_FUTURE);
			if (ret != 0) {
				fprintf(stderr, "enable_aux_rt_tasks() failed\n");
				goto out;
			}
		}
		else if (args->scheduler == RT_LINUX) {
			ret = enable_aux_rt_tasks_linux(gettid());
			if (ret != 0) {
				fprintf(stderr, "enable_aux_rt_tasks_linux() failed\n");
				goto out;
			}
		}
	}

	start = wctime();

	if (!args->gpu_using) {
		bool keepgoing;
		do
		{
			double job_ms = wcet_dist_ms->random();
			if (job_ms < 0.0)
				job_ms = 0.0;
			keepgoing = cjobfn(ms2s(job_ms * args->scale), start + args->duration);
		}while(keepgoing);
	}
	else {
		bool keepgoing;
		do
		{
			double job_ms = wcet_dist_ms->random();
			if (job_ms < 0.0)
				job_ms = 0.0;

			double cpu_job_ms = (job_ms/(args->wcet_ms + args->gpu_wcet_ms))*args->wcet_ms;
			double gpu_job_ms = (job_ms/(args->wcet_ms + args->gpu_wcet_ms))*args->gpu_wcet_ms;
			keepgoing = gjobfn(
							   ms2s(cpu_job_ms * args->scale),
							   ms2s(gpu_job_ms * args->scale),
							   args->num_kernels,
							   start + args->duration);
		}while(keepgoing);
	}

	if (args->want_signals)
		ignore_litmus_signals(SIG_BUDGET_MASK);
	

	if (args->gpu_using && ENABLE_RT_AUX_THREADS) {
		if (args->scheduler == LITMUS) {
			ret = disable_aux_rt_tasks(AUX_CURRENT | AUX_FUTURE);
			if (ret != 0) {
				fprintf(stderr, "disable_aux_rt_tasks() failed\n");
				goto out;
			}
		}
		else if(args->scheduler == RT_LINUX) {
			ret = disable_aux_rt_tasks_linux(gettid());
			if (ret != 0) {
				fprintf(stderr, "disable_aux_rt_tasks_linux() failed\n");
				goto out;
			}
		}
	}

	if (args->gpu_using)
		deallocate_locks(args->num_gpu_tasks, args->scheduler != LITMUS);

	if (args->scheduler == LITMUS)
	{
		ret = task_mode(BACKGROUND_TASK);
		if (ret != 0) {
			fprintf(stderr, "could not become regular task (huh?)\n");
			goto out;
		}
	}

	{
		// become a normal task just in case.
		struct sched_param normalparams;
		memset(&normalparams, 0, sizeof(normalparams));
		ret = sched_setscheduler(getpid(), SCHED_OTHER, &normalparams);
		if (ret < 0) {
			fprintf(stderr, "could not become sched_normal task\n");
			goto out;
		}
	}

out:
	if (wcet_dist_ms)
		delete wcet_dist_ms;

	return ret;
}

int do_normal(struct Args* args)
{
	int ret = 0;

	apply_args(args);

	if (args->scheduler == LITMUS)
		init_litmus();
	else
		init_linux();

	if (args->gpu_using) {
		signal(SIGABRT, catch_exit);
		signal(SIGTERM, catch_exit);
		signal(SIGQUIT, catch_exit);
		signal(SIGSEGV, catch_exit);

		cudaSetDeviceFlags(cudaDeviceScheduleSpin);
		init_cuda(args->num_gpu_tasks);
		init_events();
		safetynet = true;
	}

	ret = __do_normal(args);

	if (args->gpu_using) {
		safetynet = false;
		exit_cuda();
	}

	return ret;
}

typedef struct run_entry
{
	struct Args args;
	int used;
	int ret;
} run_entry_t;



static int *num_run_entries = NULL;
static run_entry_t *run_entries = NULL;
static pthread_barrier_t *daemon_barrier = NULL;
static pthread_mutex_t *daemon_mutex = NULL;

static run_entry_t *my_run_entry = NULL;
static managed_shared_memory *daemon_segment_ptr = NULL;

int init_daemon(struct Args* args, int num_total_users, bool is_daemon)
{
	if (num_total_users)
	{
		shared_memory_object::remove("gpuspin_daemon_memory");

		daemon_segment_ptr = new managed_shared_memory(create_only, "gpuspin_daemon_memory", 30*PAGE_SIZE);
		num_run_entries = daemon_segment_ptr->construct<int>("int num_run_entries")();
		*num_run_entries = num_total_users;

		run_entries = daemon_segment_ptr->construct<struct run_entry>("run_entry_t run_entries")[num_total_users]();
		memset(run_entries, 0, sizeof(run_entry_t)*num_total_users);

		daemon_mutex = daemon_segment_ptr->construct<pthread_mutex_t>("pthread_mutex_t daemon_mutex")();
		pthread_mutexattr_t attr;
		pthread_mutexattr_init(&attr);
		pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
		pthread_mutex_init(daemon_mutex, &attr);
		pthread_mutexattr_destroy(&attr);

		daemon_barrier = daemon_segment_ptr->construct<pthread_barrier_t>("pthread_barrier_t daemon_barrier")();
		pthread_barrierattr_t battr;
		pthread_barrierattr_init(&battr);
		pthread_barrierattr_setpshared(&battr, PTHREAD_PROCESS_SHARED);
		pthread_barrier_init(daemon_barrier, &battr, args->num_tasks*2);
		pthread_barrierattr_destroy(&battr);
	}
	else
	{
		do
		{
			try
			{
				if (!daemon_segment_ptr) daemon_segment_ptr = new managed_shared_memory(open_only, "gpuspin_daemon_memory");
			}
			catch(...)
			{
				sleep(1);
			}
		}while(daemon_segment_ptr == NULL);

		num_run_entries = daemon_segment_ptr->find<int>("int num_run_entries").first;
		run_entries = daemon_segment_ptr->find<struct run_entry>("run_entry_t run_entries").first;
		daemon_mutex = daemon_segment_ptr->find<pthread_mutex_t>("pthread_mutex_t daemon_mutex").first;
		daemon_barrier = daemon_segment_ptr->find<pthread_barrier_t>("pthread_barrier_t daemon_barrier").first;
	}

	if (is_daemon)
	{
		// find and claim an entry
		pthread_mutex_lock(daemon_mutex);
		for(int i = 0; i < *num_run_entries; ++i)
		{
			if(!run_entries[i].used)
			{
				my_run_entry = &run_entries[i];
				my_run_entry->used = 1;
				break;
			}
		}
		pthread_mutex_unlock(daemon_mutex);

		assert(my_run_entry);
		my_run_entry->args = *args;
		my_run_entry->ret = 0;
	}
	else
	{
		// find my entry
		pthread_mutex_lock(daemon_mutex);
		for(int i = 0; i < *num_run_entries; ++i)
		{
			if (run_entries[i].args.wcet_ms == args->wcet_ms &&
				run_entries[i].args.gpu_wcet_ms == args->gpu_wcet_ms &&
				run_entries[i].args.period_ms == args->period_ms)
			{
				my_run_entry = &run_entries[i];
				break;
			}
		}
		pthread_mutex_unlock(daemon_mutex);
	}

	if (!my_run_entry) {
		fprintf(stderr, "Could not find task <wcet, gpu_wcet, period>: <%f %f %f>\n", args->wcet_ms, args->gpu_wcet_ms, args->period_ms);
		return -1;
	}
	return 0;
}

int put_next_run(struct Args* args)
{
	assert(my_run_entry);

	pthread_mutex_lock(daemon_mutex);
	my_run_entry->args = *args;
	pthread_mutex_unlock(daemon_mutex);

	pthread_barrier_wait(daemon_barrier);

	return 0;
}

int get_next_run(struct Args* args)
{
	assert(my_run_entry);

	pthread_barrier_wait(daemon_barrier);

	pthread_mutex_lock(daemon_mutex);
	*args = my_run_entry->args;
	my_run_entry->ret = 0;
	pthread_mutex_unlock(daemon_mutex);

	return 0;
}

int complete_run(int ret)
{
	assert(my_run_entry);

	pthread_mutex_lock(daemon_mutex);
	my_run_entry->ret = ret;
	pthread_mutex_unlock(daemon_mutex);

	pthread_barrier_wait(daemon_barrier);

	return 0;
}

int wait_completion()
{
	int ret = 0;

	assert(my_run_entry);

	pthread_barrier_wait(daemon_barrier);

	pthread_mutex_lock(daemon_mutex);
	ret = my_run_entry->ret;
	pthread_mutex_unlock(daemon_mutex);

	return ret;
}




int do_proxy(struct Args* args)
{
	int ret = 0;
	ret = init_daemon(args, 0, false);
	if (ret < 0)
		goto out;
	put_next_run(args);
	ret = wait_completion();

out:
	return ret;
}

static bool is_daemon = false;
static bool running = false;
static void catch_exit2(int signal)
{
	if (is_daemon && running)
		complete_run(-signal);
	catch_exit(signal);
}

int do_daemon(struct Args* args)
{
	is_daemon = true;

	int ret = 0;
	struct Args nextargs;

	signal(SIGFPE, catch_exit2);
	signal(SIGABRT, catch_exit2);
	signal(SIGTERM, catch_exit2);
	signal(SIGQUIT, catch_exit2);
	signal(SIGSEGV, catch_exit2);

	init_daemon(args, args->num_tasks, true);

	apply_args(args);
	init_litmus(); /* does everything init_linux() does, plus litmus stuff */

	if (args->gpu_using) {
		cudaSetDeviceFlags(cudaDeviceScheduleSpin);
		init_cuda(args->num_gpu_tasks);
		init_events();
		safetynet = true;
	}

	do {
		bool sync_change = false;
		bool gpu_part_change = false;
		bool gpu_part_size_change = false;

		xprintf("%d: waiting for work\n", getpid());

		get_next_run(&nextargs);

		if (nextargs.gpu_using) {
			xprintf("%d: gpu using! gpu partition = %d, gwcet = %f, send = %lu\n",
							getpid(),
							nextargs.gpu_partition,
							nextargs.gpu_wcet_ms,
							nextargs.send_size);
		}

		running = true;
		sync_change = args->gpu_using && (CUDA_SYNC_MODE != nextargs.sync_mode);
		gpu_part_change = args->gpu_using && (GPU_PARTITION != nextargs.gpu_partition);
		gpu_part_size_change = args->gpu_using && (GPU_PARTITION_SIZE != nextargs.gpu_partition_size);

		if (sync_change || gpu_part_change || gpu_part_size_change) {
			destroy_events();
			if (gpu_part_change || gpu_part_size_change)
				exit_cuda();
		}
		apply_args(&nextargs);
		if (sync_change || gpu_part_change || gpu_part_size_change) {
			if (gpu_part_change || gpu_part_size_change) {
				xprintf("%d: changing device configuration\n", getpid());
				init_cuda(nextargs.num_gpu_tasks);
				CUR_DEVICE = -1;
				LAST_DEVICE = -1;
			}
			init_events();
		}

		xprintf("%d: starting run\n", getpid());

		ret = __do_normal(&nextargs);
		complete_run(ret);
		running = false;
	}while(ret == 0);

	if (args->gpu_using) {
		safetynet = false;
		exit_cuda();
	}

	if (args->num_gpu_tasks)
		shared_memory_object::remove("gpu_mutex_memory");

	if (args->num_tasks)
		shared_memory_object::remove("gpuspin_daemon_memory");

	return ret;
}

#define CPU_OPTIONS "p:z:c:wlveio:f:s:q:X:L:Q:d:"
#define GPU_OPTIONS "g:y:r:C:E:DG:xS:R:T:Z:aFm:b:MNIk:VW:u"
#define PROXY_OPTIONS "B:PA"

// concat the option strings
#define OPTSTR CPU_OPTIONS GPU_OPTIONS PROXY_OPTIONS

int main(int argc, char** argv)
{
	struct Args myArgs;
	set_defaults(&myArgs);

	eRunMode run_mode = NORMAL;

	int opt;

	progname = argv[0];

	while ((opt = getopt(argc, argv, OPTSTR)) != -1) {
		switch (opt) {
		case 'B':
			myArgs.num_tasks = atoi(optarg);
			break;
		case 'P':
			run_mode = PROXY;
			break;
		case 'A':
			run_mode = DAEMON;
			break;


		case 'w':
			myArgs.wait = true;
			break;
		case 'p':
			myArgs.cluster = atoi(optarg);
			myArgs.migrate = true;
			break;
		case 'z':
//			CPU_PARTITION_SIZE = cluster_size;
			myArgs.cluster_size = atoi(optarg);
			break;
		case 'g':
//			GPU_USING = true;
//			GPU_PARTITION = atoi(optarg);
			myArgs.gpu_using = true;
			myArgs.gpu_partition = atoi(optarg);
//			assert(GPU_PARTITION >= 0 && GPU_PARTITION < NR_GPUS);
			break;
		case 'y':
//			GPU_PARTITION_SIZE = atoi(optarg);
			myArgs.gpu_partition_size = atoi(optarg);
//			assert(GPU_PARTITION_SIZE > 0);
			break;
		case 'r':
//			RHO = atoi(optarg);
			myArgs.rho = atoi(optarg);
//			assert(RHO > 0);
			break;
		case 'C':
//			NUM_COPY_ENGINES = atoi(optarg);
			myArgs.num_ce = atoi(optarg);
//			assert(NUM_COPY_ENGINES == 1 || NUM_COPY_ENGINES == 2);
			break;
		case 'V':
//			RESERVED_MIGR_COPY_ENGINE = true;
			myArgs.reserve_migr_ce = true;
			break;
		case 'E':
//			USE_ENGINE_LOCKS = true;
//			ENGINE_LOCK_TYPE = (eEngineLockTypes)atoi(optarg);
			myArgs.use_engine_locks = true;
			myArgs.engine_lock_type = (eEngineLockTypes)atoi(optarg);
//			assert(ENGINE_LOCK_TYPE == FIFO || ENGINE_LOCK_TYPE == PRIOQ);
			break;
		case 'u':
			myArgs.yield_locks = true;
			break;
		case 'D':
//			USE_DYNAMIC_GROUP_LOCKS = true;
			myArgs.use_dgls = true;
			break;
		case 'G':
//			GPU_SYNC_MODE = (eGpuSyncMode)atoi(optarg);
			myArgs.gpusync_mode = (eGpuSyncMode)atoi(optarg);
//			assert(GPU_SYNC_MODE >= IKGLP_MODE && GPU_SYNC_MODE <= RGEM_MODE);
			break;
		case 'a':
//			ENABLE_AFFINITY = true;
			myArgs.enable_affinity = true;
			break;
		case 'F':
//			RELAX_FIFO_MAX_LEN = true;
			myArgs.relax_fifo_len = true;
			break;
		case 'x':
//			CUDA_SYNC_MODE = SPIN;
			myArgs.sync_mode = SPIN;
			break;
		case 'S':
//			SEND_SIZE = kbToB((size_t)atoi(optarg));
			myArgs.send_size = kbToB((size_t)atoi(optarg));
			break;
		case 'R':
//			RECV_SIZE = kbToB((size_t)atoi(optarg));
			myArgs.recv_size = kbToB((size_t)atoi(optarg));
			break;
		case 'T':
//			STATE_SIZE = kbToB((size_t)atoi(optarg));
			myArgs.state_size = kbToB((size_t)atoi(optarg));
			break;
		case 'Z':
//			ENABLE_CHUNKING = true;
//			CHUNK_SIZE = kbToB((size_t)atoi(optarg));
			myArgs.enable_chunking = true;
			myArgs.chunk_size = kbToB((size_t)atoi(optarg));
			break;
		case 'M':
//			MIGRATE_VIA_SYSMEM = true;
			myArgs.use_sysmem_migration = true;
			break;
		case 'm':
//			num_gpu_users = (int)atoi(optarg);
			myArgs.num_gpu_tasks = (int)atoi(optarg);
//			assert(num_gpu_users > 0);
			break;
		case 'k':
//			num_kernels = (unsigned int)atoi(optarg);
			myArgs.num_kernels = (unsigned int)atoi(optarg);
			break;
		case 'b':
//			budget_ms = atoi(optarg);
			myArgs.budget_ms = atoi(optarg);
			break;
		case 'W':
//			stdpct = (double)atof(optarg);
			myArgs.stddev = (double)atof(optarg);
			break;
		case 'N':
//			scheduler = LINUX;
			myArgs.scheduler = LINUX;
			break;
		case 'I':
//			scheduler = RT_LINUX;
			myArgs.scheduler = RT_LINUX;
			break;
		case 'q':
//			priority = atoi(optarg);
			myArgs.priority = atoi(optarg);
			break;
		case 'c':
//			cls = str2class(optarg);
			myArgs.cls = str2class(optarg);
			break;
		case 'e':
//			want_enforcement = true;
			myArgs.want_enforcement = true;
			break;
		case 'i':
//			want_signals = true;
			myArgs.want_signals = true;
			break;
		case 'd':
//			drain = (budget_drain_policy_t)atoi(optarg);
			myArgs.drain_policy = (budget_drain_policy_t)atoi(optarg);
//			assert(drain >= DRAIN_SIMPLE && drain <= DRAIN_SOBLIV);
//			assert(drain != DRAIN_SAWARE); // unsupported
			break;
//		case 'l':
//			test_loop = 1;
//			break;
//		case 'o':
////			column = atoi(optarg);
//			myArgs.column = atoi(optarg);
//			break;
//		case 'f':
//			file = optarg;
//			break;
		case 's':
//			scale = (double)atof(optarg);
			myArgs.scale = (double)atof(optarg);
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


	srand(time(0));

	if (argc - optind == 3) {
		myArgs.wcet_ms   = atof(argv[optind + 0]);
		myArgs.period_ms = atof(argv[optind + 1]);
		myArgs.duration  = atof(argv[optind + 2]);
	}
	else if (argc - optind == 4) {
		myArgs.wcet_ms   = atof(argv[optind + 0]);
		myArgs.gpu_wcet_ms = atof(argv[optind + 1]);
		myArgs.period_ms = atof(argv[optind + 2]);
		myArgs.duration  = atof(argv[optind + 3]);
	}

	if (myArgs.num_tasks == 0 || myArgs.num_gpu_tasks == 0) {
		// safety w.r.t. shared mem.
		sleep(2);
	}

	/* make sure children don't take sigmasks */
	ignore_litmus_signals(ALL_LITMUS_SIG_MASKS);

	if (run_mode == NORMAL) {
		return do_normal(&myArgs);
	}
	else if (run_mode == PROXY) {
		return do_proxy(&myArgs);
	}
	else if (run_mode == DAEMON) {
		return do_daemon(&myArgs);
	}
}
