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

#define xfprintf( ... ) do { \
if(!SILENT) { fprintf( __VA_ARGS__ ) ; } \
} while (0)


/* Catch errors.
 */
#define CALL( exp ) do { \
		int ret; \
		ret = exp; \
		if (ret != 0) \
			xfprintf(stderr, "%s failed: %m\n", #exp);\
		else \
			xfprintf(stderr, "%s ok.\n", #exp); \
	} while (0)

#define TH_CALL( exp ) do { \
		int ret; \
		ret = exp; \
		if (ret != 0) \
			xfprintf(stderr, "[%d] %s failed: %m\n", ctx->id, #exp); \
		else \
			xfprintf(stderr, "[%d] %s ok.\n", ctx->id, #exp); \
	} while (0)

#define TH_SAFE_CALL( exp ) do { \
		int ret; \
		xfprintf(stderr, "[%d] calling %s...\n", ctx->id, #exp); \
		ret = exp; \
		if (ret != 0) \
			xfprintf(stderr, "\t...[%d] %s failed: %m\n", ctx->id, #exp); \
		else \
			xfprintf(stderr, "\t...[%d] %s ok.\n", ctx->id, #exp); \
	} while (0)





/* these are only default values */
int NUM_THREADS=3;
int NUM_SEMS=1;
unsigned int NUM_REPLICAS=0;
int NEST_DEPTH=1;

int SILENT = 0;

int SLEEP_BETWEEN_JOBS = 1;
int USE_PRIOQ = 0;

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

#define OPTSTR "t:k:s:d:fqX"

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
			case 'q':
				USE_PRIOQ = 1;
				break;
			case 'X':
				SILENT = 1;
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
	struct rt_task param;

	struct thread_context *ctx = (struct thread_context*)_ctx;

	init_rt_task_param(&param);
	param.exec_cost = EXEC_COST;
	param.period = PERIOD + 10*ctx->id; /* Vary period a little bit. */
	param.cls = RT_CLASS_SOFT;

	TH_CALL( init_rt_thread() );
	TH_CALL( set_rt_task_param(gettid(), &param) );

	if (NUM_REPLICAS) {
		ctx->ikglp = open_ikglp_sem(ctx->fd, 0, NUM_REPLICAS);
		if(ctx->ikglp < 0)
			perror("open_ikglp_sem");
		else
			xfprintf(stdout, "ikglp od = %d\n", ctx->ikglp);
	}


	for (i = 0; i < NUM_SEMS; i++) {
		if(!USE_PRIOQ) {
			ctx->od[i] = open_fifo_sem(ctx->fd, i+1);
			if(ctx->od[i] < 0)
				perror("open_fifo_sem");
			else
				xfprintf(stdout, "fifo[%d] od = %d\n", i, ctx->od[i]);
		}
		else {
			ctx->od[i] = open_prioq_sem(ctx->fd, i+1);
			if(ctx->od[i] < 0)
				perror("open_prioq_sem");
			else
				xfprintf(stdout, "prioq[%d] od = %d\n", i, ctx->od[i]);
		}
	}

	TH_CALL( task_mode(LITMUS_RT_TASK) );


	xfprintf(stdout, "[%d] Waiting for TS release.\n ", ctx->id);
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


		if(NUM_REPLICAS) {
			replica = litmus_lock(ctx->ikglp);
			xfprintf(stdout, "[%d] got ikglp replica %d.\n", ctx->id, replica);
		}


		litmus_dgl_lock(dgl, dgl_size);
		xfprintf(stdout, "[%d] acquired dgl.\n", ctx->id);

		do_exit = job(ctx);


		xfprintf(stdout, "[%d] unlocking dgl.\n", ctx->id);
		litmus_dgl_unlock(dgl, dgl_size);

		if(NUM_REPLICAS) {
			xfprintf(stdout, "[%d]: freeing ikglp replica %d.\n", ctx->id, replica);
			litmus_unlock(ctx->ikglp);
		}

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
