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


#define NUM_CPUS	4

//#define NUM_THREADS	3
int NUM_THREADS=3;

/* NEST_DEPTH may not be greater than NUM_SEMS. */
//#define NUM_SEMS	10
int NUM_SEMS=10;

int SLEEP_BETWEEN_JOBS = 1;

#define MAX_SEMS 1000

//#define NEST_DEPTH	 5
int NEST_DEPTH=5;

#define EXEC_COST 	 1000*1
#define PERIOD		1000*10

/* The information passed to each thread. Could be anything. */
struct thread_context {
	int id;
	int fd;
	int od[MAX_SEMS];
	int count;
	unsigned int rand;
};

void* rt_thread(void* _ctx);
int nested_job(struct thread_context* ctx, int *count, int *next);
int job(struct thread_context*);

#define OPTSTR "t:s:d:f"

int main(int argc, char** argv)
{
	int i;
	struct thread_context* ctx; //[NUM_THREADS];
	pthread_t*	     task;  //[NUM_THREADS];
	int fd;

	int opt;
	while((opt = getopt(argc, argv, OPTSTR)) != -1) {
		switch(opt) {
			case 't':
				NUM_THREADS = atoi(optarg);
				break;
			case 's':
				NUM_SEMS = atoi(optarg);
				assert(NUM_SEMS <= MAX_SEMS);
				break;
			case 'd':
				NEST_DEPTH = atoi(optarg);
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

	/* Make presence visible. */
	//printf("RT Thread %d active.\n", ctx->id);

	TH_CALL( init_rt_thread() );
	TH_CALL( sporadic_task_ns(EXEC_COST, PERIOD + 10*ctx->id, 0, 0,
		LITMUS_LOWEST_PRIORITY, RT_CLASS_SOFT, NO_ENFORCEMENT, NO_SIGNALS, 0) );

	for (i = 0; i < NUM_SEMS; i++) {
		ctx->od[i] = open_rsm_sem(ctx->fd, i);
		if(ctx->od[i] < 0)
			perror("open_rsm_sem");
		//printf("[%d] ctx->od[%d]: %d\n", ctx->id, i, ctx->od[i]);
	}

	TH_CALL( task_mode(LITMUS_RT_TASK) );


	printf("[%d] Waiting for TS release.\n ", ctx->id);
	wait_for_ts_release();
	ctx->count = 0;

	do {
		int first = (int)(NUM_SEMS * (rand_r(&(ctx->rand)) / (RAND_MAX + 1.0)));
		int count = NEST_DEPTH;
		do_exit = nested_job(ctx, &count, &first);
		
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


int nested_job(struct thread_context* ctx, int *count, int *next)
{
	int ret;

	if(*count == 0 || *next == NUM_SEMS)  /* base case */
	{
		ret = job(ctx);
	}
	else
	{
		int which_sem = ctx->od[*next];

		++(*next);
		--(*count);

		printf("[%d]: trying to get semaphore %d.\n", ctx->id, which_sem);
		fflush(stdout);
		TH_SAFE_CALL ( litmus_lock(which_sem) );
		printf("[%d] got semaphore %d.\n", ctx->id, which_sem);
		fflush(stdout);
		ret = nested_job(ctx, count, next);
		TH_SAFE_CALL ( litmus_unlock(which_sem) );
		fflush(stdout);
	}

	return(ret);
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
	return ctx->count++ > 30000;
}
