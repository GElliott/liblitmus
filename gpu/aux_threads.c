/* based_mt_task.c -- A basic multi-threaded real-time task skeleton.
 *
 * This (by itself useless) task demos how to setup a multi-threaded LITMUS^RT
 * real-time task. Familiarity with the single threaded example (base_task.c)
 * is assumed.
 *
 * Currently, liblitmus still lacks automated support for real-time
 * tasks, but internaly it is thread-safe, and thus can be used together
 * with pthreads.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/resource.h>

/* Include gettid() */
#include <sys/types.h>

/* Include threading support. */
#include <pthread.h>

/* Include the LITMUS^RT API.*/
#include "litmus.h"

//#define PERIOD		500
#define PERIOD		 10
//#define EXEC_COST	 10
#define EXEC_COST 1

int NUM_AUX_THREADS = 2;

#define LITMUS_STATS_FILE "/proc/litmus/stats"

/* The information passed to each thread. Could be anything. */
struct thread_context {
	int id;
	struct timeval total_time;
};

/* The real-time thread program. Doesn't have to be the same for
 * all threads. Here, we only have one that will invoke job().
 */
void* rt_thread(void *tcontext);
void* aux_thread(void *tcontext);

/* Declare the periodically invoked job.
 * Returns 1 -> task should exit.
 *         0 -> task should continue.
 */
int job(void);


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

int gRun = 1;

pthread_mutex_t gMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_barrier_t gBar;

#define OPTSTR "t:fcb"

int main(int argc, char** argv)
{
	int i;
	struct thread_context *ctx;
	pthread_t *task;

	int opt;
	int before = 0;
	int aux_flags = 0;
	int do_future = 0;

	while ((opt = getopt(argc, argv, OPTSTR)) != -1) {
		switch(opt)
		{
		case 't':
			NUM_AUX_THREADS = atoi(optarg);
			printf("%d aux threads\n", NUM_AUX_THREADS);
			break;
		case 'f':
			aux_flags |= AUX_FUTURE;
			do_future = 1;
			break;
		case 'c':
			aux_flags |= AUX_CURRENT;
			break;
		case 'b':
			before = 1;
			printf("Will become real-time before spawning aux threads.\n");
			break;
		}
	}

	if (aux_flags == 0) {
		printf("Must specify -c (AUX_CURRENT) and/or -f (AUX_FUTURE) for aux tasks.\n");
		return -1;
	}

	ctx = calloc(NUM_AUX_THREADS, sizeof(struct thread_context));
	task = calloc(NUM_AUX_THREADS, sizeof(pthread_t));

	//lt_t delay = ms2lt(1000);

	/*****
	 * 3) Initialize LITMUS^RT.
	 *    Task parameters will be specified per thread.
	 */
	init_litmus();

	{
		pthread_barrierattr_t battr;
		pthread_barrierattr_init(&battr);
		pthread_barrier_init(&gBar, &battr, (NUM_AUX_THREADS)+1);
	}

	if(before)
	{
		CALL( init_rt_thread() );
		CALL( sporadic_partitioned(EXEC_COST, PERIOD, 0) );
		CALL( task_mode(LITMUS_RT_TASK) );
	}


	if(do_future && before)
	{
		CALL( enable_aux_rt_tasks(aux_flags) );
	}

//	printf("Red Leader is now real-time!\n");

	for (i = 0; i < NUM_AUX_THREADS; i++) {
		ctx[i].id = i;
		pthread_create(task + i, NULL, aux_thread, (void *) (ctx + i));
	}

//	pthread_barrier_wait(&gBar);

//	sleep(1);

	if(!before)
	{
		CALL( init_rt_thread() );
		CALL( sporadic_global(EXEC_COST, PERIOD) );
		CALL( task_mode(LITMUS_RT_TASK) );
	}

	// secondary call *should* be harmless
	CALL( enable_aux_rt_tasks(aux_flags) );

	{
	int last = time(0);
//	struct timespec sleeptime = {0, 1000}; // 1 microsecond
//	for(i = 0; i < 24000; ++i) {
	for(i = 0; i < 2000; ++i) {
		sleep_next_period();
//		printf("RED LEADER!\n");

//		nanosleep(&sleeptime, NULL);

		pthread_mutex_lock(&gMutex);

		if((i%(10000/PERIOD)) == 0) {
			int now = time(0);
			printf("hearbeat %d: %d\n", i, now - last);
			last = now;
		}

		pthread_mutex_unlock(&gMutex);
	}
	}

	CALL( disable_aux_rt_tasks(aux_flags) );
	gRun = 0;

	CALL( task_mode(BACKGROUND_TASK) );

	/*****
	 * 5) Wait for RT threads to terminate.
	 */
	for (i = 0; i < NUM_AUX_THREADS; i++) {
		if (task[i] != 0) {
			float time;
			pthread_join(task[i], NULL);
			time = ctx[i].total_time.tv_sec + ctx[i].total_time.tv_usec / (float)(1e6);
			printf("child %d: %fs\n", i, time);
		}
	}


	/*****
	 * 6) Clean up, maybe print results and stats, and exit.
	 */
	return 0;
}



/* A real-time thread is very similar to the main function of a single-threaded
 * real-time app. Notice, that init_rt_thread() is called to initialized per-thread
 * data structures of the LITMUS^RT user space libary.
 */
void* aux_thread(void *tcontext)
{
	struct thread_context *ctx = (struct thread_context *) tcontext;
	int count = 0;

//	pthread_barrier_wait(&gBar);

	while(gRun)
	{
		if(count++ % 100000 == 0) {
			pthread_mutex_lock(&gMutex);
			pthread_mutex_unlock(&gMutex);
		}
	}

	{
	struct rusage use;
	long int sec;

	getrusage(RUSAGE_THREAD, &use);

	ctx->total_time.tv_usec = use.ru_utime.tv_usec + use.ru_stime.tv_usec;
	sec = ctx->total_time.tv_usec / (long int)(1e6);
	ctx->total_time.tv_usec = ctx->total_time.tv_usec % (long int)(1e6);
	ctx->total_time.tv_sec = use.ru_utime.tv_sec + use.ru_stime.tv_sec + sec;
	}

	return ctx;
}


/* A real-time thread is very similar to the main function of a single-threaded
 * real-time app. Notice, that init_rt_thread() is called to initialized per-thread
 * data structures of the LITMUS^RT user space libary.
 */
void* rt_thread(void *tcontext)
{
	struct thread_context *ctx = (struct thread_context *) tcontext;

	/* Make presence visible. */
	printf("RT Thread %d active.\n", ctx->id);

	/*****
	 * 1) Initialize real-time settings.
	 */
	CALL( init_rt_thread() );
	CALL( sporadic_global(EXEC_COST, PERIOD + ctx->id * 10) );


	/*****
	 * 2) Transition to real-time mode.
	 */
	CALL( task_mode(LITMUS_RT_TASK) );



	wait_for_ts_release();

	/* The task is now executing as a real-time task if the call didn't fail.
	 */



	/*****
	 * 3) Invoke real-time jobs.
	 */
	while(gRun) {
		/* Wait until the next job is released. */
		sleep_next_period();
		printf("%d: task.\n", ctx->id);
	}

	/*****
	 * 4) Transition to background mode.
	 */
	CALL( task_mode(BACKGROUND_TASK) );

	{
	struct rusage use;
	long int sec;

	getrusage(RUSAGE_THREAD, &use);
	ctx->total_time.tv_usec = use.ru_utime.tv_usec + use.ru_stime.tv_usec;
	sec = ctx->total_time.tv_usec / (long int)(1e6);
	ctx->total_time.tv_usec = ctx->total_time.tv_usec % (long int)(1e6);
	ctx->total_time.tv_sec = use.ru_utime.tv_sec + use.ru_stime.tv_sec + sec;
	}

	return ctx;
}

int job(void)
{
	/* Do real-time calculation. */

	/* Don't exit. */
	return 0;
}
