#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <assert.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

/* Include gettid() */
#include <sys/types.h>

/* Include threading support. */
#include <pthread.h>

/* Include the LITMUS^RT API.*/
#include "litmus.h"

/* Catch errors.
 */
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


/* these are only default values */
int NUM_THREADS=3;
int NUM_SEMS=1;
int NUM_REPLICAS=1;
int NEST_DEPTH=1;

int SLEEP_BETWEEN_JOBS = 1;

#define MAX_SEMS 1000
#define MAX_NEST_DEPTH 10


// 1000 = 1us
#define EXEC_COST 	 1000*1
#define PERIOD		1000*10

/* The information passed to each thread. Could be anything. */
struct thread_context {
	int id;
	int fd;
	int ikglp;
	int od[MAX_SEMS];
	int count;
	unsigned int rand;
};

void* rt_thread(void* _ctx);
int nested_job(struct thread_context* ctx, int *count, int *next);
int job(struct thread_context*);

#define OPTSTR "t:k:s:d:f"

int main(int argc, char** argv)
{
	int i;
	struct thread_context* ctx;
	pthread_t*	     task;
	int fd;

	int opt;
	while((opt = getopt(argc, argv, OPTSTR)) != -1) {
		switch(opt) {
			case 't':
				NUM_THREADS = atoi(optarg);
				break;
			case 'k':
				NUM_REPLICAS = atoi(optarg);
				assert(NUM_REPLICAS > 0);
				break;
			case 's':
				NUM_SEMS = atoi(optarg);
				assert(NUM_SEMS >= 0 && NUM_SEMS <= MAX_SEMS);
				break;
			case 'd':
				NEST_DEPTH = atoi(optarg);
				assert(NEST_DEPTH >= 1 && NEST_DEPTH <= MAX_NEST_DEPTH);
				break;
			case 'f':
				SLEEP_BETWEEN_JOBS = 0;
				break;
			default:
				fprintf(stderr, "Unknown option: %c\n", opt);
				exit(-1);
				break;
		}
	}

	ctx = (struct thread_context*) calloc(NUM_THREADS, sizeof(struct thread_context));
	task = (pthread_t*) calloc(NUM_THREADS, sizeof(pthread_t));

	srand(0); /* something repeatable for now */

	fd = open("semaphores", O_RDONLY | O_CREAT, S_IRUSR | S_IWUSR);

	CALL( init_litmus() );

	for (i = 0; i < NUM_THREADS; i++) {
		ctx[i].id = i;
		ctx[i].fd = fd;
		ctx[i].rand = rand();
		CALL( pthread_create(task + i, NULL, rt_thread, ctx + i) );
	}


	for (i = 0; i < NUM_THREADS; i++)
		pthread_join(task[i], NULL);


	return 0;
}

void* rt_thread(void* _ctx)
{
	int i;
	int do_exit = 0;

	struct thread_context *ctx = (struct thread_context*)_ctx;

	TH_CALL( init_rt_thread() );

	/* Vary period a little bit. */
	TH_CALL( sporadic_task_ns(EXEC_COST, PERIOD + 10*ctx->id, 0, 0, RT_CLASS_SOFT, NO_ENFORCEMENT, NO_SIGNALS, 0) );

	ctx->ikglp = open_ikglp_sem(ctx->fd, 0, (void*)&NUM_REPLICAS);
	if(ctx->ikglp < 0)
		perror("open_ikglp_sem");
	else
		printf("ikglp od = %d\n", ctx->ikglp);

	for (i = 0; i < NUM_SEMS; i++) {
		ctx->od[i] = open_rsm_sem(ctx->fd, i+1);
		if(ctx->od[i] < 0)
			perror("open_rsm_sem");
		else
			printf("rsm[%d] od = %d\n", i, ctx->od[i]);
	}

	TH_CALL( task_mode(LITMUS_RT_TASK) );


	printf("[%d] Waiting for TS release.\n ", ctx->id);
	wait_for_ts_release();
	ctx->count = 0;

	do {
		int replica = -1;
		int first = (int)(NUM_SEMS * (rand_r(&(ctx->rand)) / (RAND_MAX + 1.0)));
		int last = (first + NEST_DEPTH - 1 >= NUM_SEMS) ? NUM_SEMS - 1 : first + NEST_DEPTH - 1;
		int dgl_size = last - first + 1;
		int dgl[dgl_size];
		
		// construct the DGL
		for(i = first; i <= last; ++i) {
			dgl[i-first] = ctx->od[i];
		}
		
		
		replica = litmus_lock(ctx->ikglp);
		printf("[%d] got ikglp replica %d.\n", ctx->id, replica);
		fflush(stdout);

		
		litmus_dgl_lock(dgl, dgl_size);
		printf("[%d] acquired dgl.\n", ctx->id);
		fflush(stdout);
		
		
		do_exit = job(ctx);

		
		printf("[%d] unlocking dgl.\n", ctx->id);
		fflush(stdout);		
		litmus_dgl_unlock(dgl, dgl_size);
		
		
		printf("[%d]: freeing ikglp replica %d.\n", ctx->id, replica);
		fflush(stdout);
		litmus_unlock(ctx->ikglp);

		if(SLEEP_BETWEEN_JOBS && !do_exit) {
			sleep_next_period();
		}
	} while(!do_exit);

	/*****
	 * 4) Transition to background mode.
	 */
	TH_CALL( task_mode(BACKGROUND_TASK) );


	return NULL;
}

void dirty_kb(int kb) 
{
	int32_t one_kb[256];
	int32_t sum = 0;
	int32_t i;

	for (i = 0; i < 256; i++)
		sum += one_kb[i];
	kb--;
	/* prevent tail recursion */
	if (kb)
		dirty_kb(kb);
	for (i = 0; i < 256; i++)
		sum += one_kb[i];
}

int job(struct thread_context* ctx)
{
	/* Do real-time calculation. */
	dirty_kb(8);

	/* Don't exit. */
	//return ctx->count++ > 100;
	//return ctx->count++ > 12000;
	//return ctx->count++ > 120000;
	return ctx->count++ >   50000;  // controls number of jobs per task
}
