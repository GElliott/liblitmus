#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>

/* Include gettid() */
#include <sys/types.h>

/* Include threading support. */
#include <pthread.h>

/* Include the LITMUS^RT API.*/
#include "litmus.h"

/* Catch errors.
 */
#if 1
#define CALL( exp ) do { \
		int ret; \
		ret = exp; \
		if (ret != 0) \
			fprintf(stderr, "%s failed: %m\n", #exp);\
		else \
			fprintf(stderr, "%s ok.\n", #exp); \
	} while (0)

#define TH_CALL( exp ) do { \
		int ret; \
		ret = exp; \
		if (ret != 0) \
			fprintf(stderr, "[%d] %s failed: %m\n", ctx->id, #exp); \
		else \
			fprintf(stderr, "[%d] %s ok.\n", ctx->id, #exp); \
	} while (0)

#define TH_SAFE_CALL( exp ) do { \
		int ret; \
		fprintf(stderr, "[%d] calling %s...\n", ctx->id, #exp); \
		ret = exp; \
		if (ret != 0) \
			fprintf(stderr, "\t...[%d] %s failed: %m\n", ctx->id, #exp); \
		else \
			fprintf(stderr, "\t...[%d] %s ok.\n", ctx->id, #exp); \
	} while (0)
#else
#define CALL( exp )
#define TH_CALL( exp )
#define TH_SAFE_CALL( exp )
#endif

/* these are only default values */
int NUM_THREADS=3;
int NUM_AUX_THREADS=0;
int NUM_SEMS=1;
int NUM_GPUS=1;
int GPU_OFFSET=0;
int NUM_SIMULT_USERS = 1;
int ENABLE_AFFINITY = 0;
int NEST_DEPTH=1;
int USE_KFMLP = 0;
int RELAX_FIFO_MAX_LEN = 0;
int USE_DYNAMIC_GROUP_LOCKS = 0;

int SLEEP_BETWEEN_JOBS = 1;
int USE_PRIOQ = 0;

int gAuxRun = 1;
pthread_mutex_t gMutex = PTHREAD_MUTEX_INITIALIZER;

#define MAX_SEMS 1000

// 1000 = 1us
#define EXEC_COST 	1000*1
#define PERIOD		2*1000*100

/* The information passed to each thread. Could be anything. */
struct thread_context {
	int id;
	int fd;
	int kexclu;
	int od[MAX_SEMS];
	int count;
	unsigned int rand;
	int mig_count[5];
};

void* rt_thread(void* _ctx);
void* aux_thread(void* _ctx);
int nested_job(struct thread_context* ctx, int *count, int *next, int runfactor);
int job(struct thread_context* ctx, int runfactor);


struct avg_info
{
	float avg;
	float stdev;
};

struct avg_info feedback(int _a, int _b)
{
	fp_t a = _frac(_a, 10000);
	fp_t b = _frac(_b, 10000);
	int i;

	fp_t actual_fp;

	fp_t _est, _err;

	int base = 1000000;
	//int range = 40;

	fp_t est = _integer_to_fp(base);
	fp_t err = _fp(base/2);

#define NUM_SAMPLES 10000

	float samples[NUM_SAMPLES] = {0.0};
	float accu_abs, accu;
	float avg;
	float devsum;
	float stdev;
	struct avg_info ret;

	for(i = 0; i < NUM_SAMPLES; ++i) {
		int num = ((rand()%40)*(rand()%2 ? -1 : 1)/100.0)*base + base;
		float rel_err;

		actual_fp = _integer_to_fp(num);

//	printf("Before: est = %d\terr = %d\n", (int)_fp_to_integer(est), (int)_fp_to_integer(err));

		_err = _sub(actual_fp, est);
		_est = _add(_mul(a, _err), _mul(b, err));

		rel_err = _fp_to_integer(_mul(_div(_err, est), _integer_to_fp(10000)))/10000.0;
		rel_err *= 100.0;
		//printf("%6.2f\n", rel_err);
		samples[i] = rel_err;

		est = _est;
		err = _add(err, _err);

		if((int)_fp_to_integer(est) <= 0) {
			est = actual_fp;
			err = _div(actual_fp, _integer_to_fp(2));
		}

	//printf("After: est = %d\terr = %d\n", (int)_fp_to_integer(est), (int)_fp_to_integer(err));
	}

	accu_abs = 0.0;
	accu = 0.0;
	for(i = 0; i < NUM_SAMPLES; ++i) {
		accu += samples[i];
		accu_abs += abs(samples[i]);
	}

	avg = accu_abs/NUM_SAMPLES;
	devsum = 0;
	for(i = 0; i < NUM_SAMPLES; ++i) {
		float dev = samples[i] - avg;
		dev *= dev;
		devsum += dev;
	}

	stdev = sqrtf(devsum/(NUM_SAMPLES-1));
	
	ret.avg = avg;
	ret.stdev = stdev;

	//printf("AVG: %6.2f\tw/ neg: %6.2f\n", accu_abs/NUM_SAMPLES, accu/NUM_SAMPLES);

	//return (accu_abs/NUM_SAMPLES);
	return(ret);
}



#define OPTSTR "t:k:o:z:s:d:lfaryA:q"

int main(int argc, char** argv)
{
	int i;
	struct thread_context* ctx;
	struct thread_context* aux_ctx;
	pthread_t*	     task;
	pthread_t*	     aux_task;
	int fd;

	int opt;
	while((opt = getopt(argc, argv, OPTSTR)) != -1) {
		switch(opt) {
			case 't':
				NUM_THREADS = atoi(optarg);
				break;
			case 'A':
				NUM_AUX_THREADS = atoi(optarg);
				break;
			case 'k':
				NUM_GPUS = atoi(optarg);
				assert(NUM_GPUS > 0);
				break;
			case 'z':
				NUM_SIMULT_USERS = atoi(optarg);
				assert(NUM_SIMULT_USERS > 0);
				break;
			case 'o':
				GPU_OFFSET = atoi(optarg);
				assert(GPU_OFFSET >= 0);
				break;
			case 's':
				NUM_SEMS = atoi(optarg);
				assert(NUM_SEMS >= 0 && NUM_SEMS < MAX_SEMS);
				break;
			case 'd':
				NEST_DEPTH = atoi(optarg);
				assert(NEST_DEPTH >= 0);
				break;
			case 'f':
				SLEEP_BETWEEN_JOBS = 0;
				break;
			case 'a':
				ENABLE_AFFINITY = 1;
				break;
			case 'l':
				USE_KFMLP = 1;
				break;
			case 'y':
				USE_DYNAMIC_GROUP_LOCKS = 1;
				break;
			case 'r':
				RELAX_FIFO_MAX_LEN = 1;
				break;
			case 'q':
				USE_PRIOQ = 1;
				break;
			default:
				fprintf(stderr, "Unknown option: %c\n", opt);
				exit(-1);
				break;
		}
	}

#if 0
	int best_a = 0, best_b = 0;
	int first = 1;
	int TRIALS = 15;

	int a, b, t;

	struct avg_info best = {0.0,0.0}, second_best;

	int second_best_a, second_best_b;

	srand(time(0));

	int step = 50;

	for(b = 2000; b < 5000; b += step) {
		for(a = 1500; a < b; a += (step/4)) {
			float std_accum = 0;
			float avg_accum = 0;
			for(t = 0; t < TRIALS; ++t) {
				struct avg_info temp;
				temp = feedback(a, b);
				std_accum += temp.stdev;
				avg_accum += temp.avg;
			}

			float avg_std = std_accum / TRIALS;

			if(first || avg_std < best.stdev) {
				second_best_a = best_a;
				second_best_b = best_b;
				second_best = best;

				best.stdev = avg_std;
				best.avg = avg_accum / TRIALS;
				best_a = a;
				best_b = b;

				first = 0;
			}
		}
	}
	
	printf("Best:\ta = %d\tb = %d\t(b-a) = %d\tavg = %6.2f\tstdev = %6.2f\n", best_a, best_b, best_b - best_a, best.avg, best.stdev);
	printf("2nd:\ta = %d\tb = %d\t(b-a) = %d\tavg = %6.2f\tstdev = %6.2f\n", second_best_a, second_best_b, second_best_b - second_best_a, second_best.avg, second_best.stdev);


			a = 14008;
			b = 16024;
			float std_accum = 0;
			float avg_accum = 0;
			for(t = 0; t < TRIALS; ++t) {
				struct avg_info temp;
				temp = feedback(a, b);
				std_accum += temp.stdev;
				avg_accum += temp.avg;
			}

	printf("Aaron:\tavg = %6.2f\tstd = %6.2f\n", avg_accum/TRIALS, std_accum/TRIALS);
	



	return 0;
#endif




	ctx = (struct thread_context*) calloc(NUM_THREADS, sizeof(struct thread_context));
	task = (pthread_t*) calloc(NUM_THREADS, sizeof(pthread_t));

	if (NUM_AUX_THREADS) {
		aux_ctx = (struct thread_context*) calloc(NUM_AUX_THREADS, sizeof(struct thread_context));
		aux_task = (pthread_t*) calloc(NUM_AUX_THREADS, sizeof(pthread_t));
	}

	srand(0); /* something repeatable for now */

	fd = open("semaphores", O_RDONLY | O_CREAT, S_IRUSR | S_IWUSR);

	CALL( init_litmus() );

	for (i = 0; i < NUM_AUX_THREADS; i++) {
		aux_ctx[i].id = i;
		CALL( pthread_create(aux_task + i, NULL, aux_thread, ctx + i) );
	}

	for (i = 0; i < NUM_THREADS; i++) {
		ctx[i].id = i;
		ctx[i].fd = fd;
		ctx[i].rand = rand();
		memset(&ctx[i].mig_count, 0, sizeof(ctx[i].mig_count));
		CALL( pthread_create(task + i, NULL, rt_thread, ctx + i) );
	}

	if (NUM_AUX_THREADS) {
		TH_CALL( init_rt_thread() );
		TH_CALL( sporadic_task_ns(EXEC_COST, PERIOD + 10*NUM_THREADS+1, 0, 0,
			LITMUS_LOWEST_PRIORITY, RT_CLASS_SOFT, NO_ENFORCEMENT, NO_SIGNALS, 1) );
		TH_CALL( task_mode(LITMUS_RT_TASK) );

		printf("[MASTER] Waiting for TS release.\n ");
		wait_for_ts_release();

		CALL( enable_aux_rt_tasks(AUX_CURRENT) );

		for(i = 0; i < 25000; ++i) {
			sleep_next_period();
			pthread_mutex_lock(&gMutex);
			pthread_mutex_unlock(&gMutex);
		}

		CALL( disable_aux_rt_tasks(AUX_CURRENT) );
		__sync_synchronize();
		gAuxRun = 0;
		__sync_synchronize();

		for (i = 0; i < NUM_AUX_THREADS; i++)
			pthread_join(aux_task[i], NULL);

		TH_CALL( task_mode(BACKGROUND_TASK) );
	}

	for (i = 0; i < NUM_THREADS; i++)
		pthread_join(task[i], NULL);

	return 0;
}

int affinity_cost[] = {1, 4, 8, 16};

int affinity_distance(struct thread_context* ctx, int a, int b)
{
	int i;
	int dist;
	
	if(a >= 0 && b >= 0) {
		for(i = 0; i <= 3; ++i) {
			if(a>>i == b>>i) {
				dist = i;
				goto out;
			}
		}
		dist = 0; // hopefully never reached.
	}
	else {
		dist = 0;
	}	
	
out:
	//printf("[%d]: distance: %d -> %d = %d\n", ctx->id, a, b, dist);	
	
	++(ctx->mig_count[dist]);
	
	return dist;
	
//	int groups[] = {2, 4, 8};
//	int i;
//	
//	if(a < 0 || b < 0)
//		return (sizeof(groups)/sizeof(groups[0]));  // worst affinity
//	
//	// no migration
//	if(a == b)
//		return 0;
//	
//	for(i = 0; i < sizeof(groups)/sizeof(groups[0]); ++i) {
//		if(a/groups[i] == b/groups[i])
//			return (i+1);
//	}
//	assert(0);
//	return -1;
}


void* aux_thread(void* _ctx)
{
	struct thread_context *ctx = (struct thread_context*)_ctx;

	while (gAuxRun) {
		pthread_mutex_lock(&gMutex);
		pthread_mutex_unlock(&gMutex);
	}

	return ctx;
}

void* rt_thread(void* _ctx)
{
	int i;
	int do_exit = 0;
	int last_replica = -1;	

	struct thread_context *ctx = (struct thread_context*)_ctx;

	TH_CALL( init_rt_thread() );

	/* Vary period a little bit. */
	TH_CALL( sporadic_task_ns(EXEC_COST, PERIOD + 10*ctx->id, 0, 0,
		LITMUS_LOWEST_PRIORITY, RT_CLASS_SOFT, NO_ENFORCEMENT, NO_SIGNALS, 1) );

	if(USE_KFMLP) {
		ctx->kexclu = open_kfmlp_gpu_sem(ctx->fd,
										 0,  /* name */
										 NUM_GPUS,
										 GPU_OFFSET,
										 NUM_SIMULT_USERS,
										 ENABLE_AFFINITY
										 );
	}
	else {
//		ctx->kexclu = open_ikglp_sem(ctx->fd, 0, &NUM_GPUS);
		ctx->kexclu = open_gpusync_token_lock(ctx->fd,
								0,  /* name */
								NUM_GPUS,
								GPU_OFFSET,
								NUM_SIMULT_USERS,
								IKGLP_M_IN_FIFOS,
								(!RELAX_FIFO_MAX_LEN) ?
									  IKGLP_OPTIMAL_FIFO_LEN :
									  IKGLP_UNLIMITED_FIFO_LEN,
								ENABLE_AFFINITY
								);	
	}
	if(ctx->kexclu < 0)
		perror("open_kexclu_sem");
	else
		printf("kexclu od = %d\n", ctx->kexclu);
	
	for (i = 0; i < NUM_SEMS; ++i) {
		if(!USE_PRIOQ) {
			ctx->od[i] = open_fifo_sem(ctx->fd, i + ctx->kexclu + 2);
			if(ctx->od[i] < 0)
				perror("open_fifo_sem");
			else
				printf("fifo[%d] od = %d\n", i, ctx->od[i]);
		}
		else {
			ctx->od[i] = open_prioq_sem(ctx->fd, i + ctx->kexclu + 2);
			if(ctx->od[i] < 0)
				perror("open_prioq_sem");
			else
				printf("prioq[%d] od = %d\n", i, ctx->od[i]);
		}
	}

	TH_CALL( task_mode(LITMUS_RT_TASK) );

	printf("[%d] Waiting for TS release.\n ", ctx->id);
	wait_for_ts_release();
	ctx->count = 0;

	do {
		int first = (int)(NUM_SEMS * (rand_r(&(ctx->rand)) / (RAND_MAX + 1.0)));
		int last = (first + NEST_DEPTH - 1 >= NUM_SEMS) ? NUM_SEMS - 1 : first + NEST_DEPTH - 1;
		int dgl_size = last - first + 1;
		int replica = -1;
		int distance;
		
		int dgl[dgl_size];		
		
		// construct the DGL
		for(i = first; i <= last; ++i) {
			dgl[i-first] = ctx->od[i];
		}		
		
		replica = litmus_lock(ctx->kexclu);

		//printf("[%d] got kexclu replica %d.\n", ctx->id, replica);
		//fflush(stdout);

		distance = affinity_distance(ctx, replica, last_replica);
		
		if(USE_DYNAMIC_GROUP_LOCKS) {
			litmus_dgl_lock(dgl, dgl_size);
		}
		else {
			for(i = 0; i < dgl_size; ++i) {
				litmus_lock(dgl[i]);
			}
		}
		
		//do_exit = nested_job(ctx, &count, &first, affinity_cost[distance]);
		do_exit = job(ctx, affinity_cost[distance]);
		
		if(USE_DYNAMIC_GROUP_LOCKS) {
			litmus_dgl_unlock(dgl, dgl_size);
		}
		else {
			for(i = dgl_size - 1; i >= 0; --i) {
				litmus_unlock(dgl[i]);
			}			
		}		
		
		//printf("[%d]: freeing kexclu replica %d.\n", ctx->id, replica);
		//fflush(stdout);

		litmus_unlock(ctx->kexclu);
		
		last_replica = replica;

		if(SLEEP_BETWEEN_JOBS && !do_exit) {
			sleep_next_period();
		}
	} while(!do_exit);

//	if (ctx->id == 0 && NUM_AUX_THREADS) {
//		gAuxRun = 0;
//		__sync_synchronize();
//		CALL( disable_aux_rt_tasks() );
//	}

	/*****
	 * 4) Transition to background mode.
	 */
	TH_CALL( task_mode(BACKGROUND_TASK) );

	for(i = 0; i < sizeof(ctx->mig_count)/sizeof(ctx->mig_count[0]); ++i) 
	{
		printf("[%d]: mig_count[%d] = %d\n", ctx->id, i, ctx->mig_count[i]);
	}

	return NULL;
}

//int nested_job(struct thread_context* ctx, int *count, int *next, int runfactor)
//{
//	int ret;
//
//	if(*count == 0 || *next == NUM_SEMS)
//	{
//		ret = job(ctx, runfactor);
//	}
//	else
//	{
//		int which_sem = *next;
//		int rsm_od = ctx->od[which_sem];
//
//		++(*next);
//		--(*count);
//
//		//printf("[%d]: trying to get semaphore %d.\n", ctx->id, which_sem);
//		//fflush(stdout);
//		litmus_lock(rsm_od);
//
//		//printf("[%d] got semaphore %d.\n", ctx->id, which_sem);
//		//fflush(stdout);
//		ret = nested_job(ctx, count, next, runfactor);
//
//		//printf("[%d]: freeing semaphore %d.\n", ctx->id, which_sem);
//		//fflush(stdout);
//		litmus_unlock(rsm_od);
//	}
//
//return(ret);
//}


void dirty_kb(int kb) 
{	
	int32_t one_kb[256];
	int32_t sum = 0;
	int32_t i;

	if(!kb)
		return;	
	
	for (i = 0; i < 256; i++)
		sum += one_kb[i];
	kb--;
	/* prevent tail recursion */
	if (kb)
		dirty_kb(kb);
	for (i = 0; i < 256; i++)
		sum += one_kb[i];
}

int job(struct thread_context* ctx, int runfactor)
{
	//struct timespec tosleep = {0, 100000}; // 0.1 ms
	
	//printf("[%d]: runfactor = %d\n", ctx->id, runfactor);
	
	//dirty_kb(8 * runfactor);
	dirty_kb(1 * runfactor);
	//nanosleep(&tosleep, NULL);

	/* Don't exit. */
	//return ctx->count++ > 100;
	//return ctx->count++ > 12000;
	//return ctx->count++ > 120000;
	return ctx->count++ >   25000;  // controls number of jobs per task
}
