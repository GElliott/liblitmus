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
// 1000 = 1us
#define EXEC_COST 	1000*1
#define PERIOD		2*1000*100


int main(int argc, char** argv)
{
	CALL( init_litmus() );

	CALL( init_rt_thread() );
	CALL( sporadic_task_ns(EXEC_COST, PERIOD, 0, 0,
		LITMUS_LOWEST_PRIORITY, RT_CLASS_SOFT, NO_ENFORCEMENT, NO_SIGNALS, 1) );
	//CALL( task_mode(LITMUS_RT_TASK) );

	fprintf(stdout, "Waiting for TS release.\n ");
	wait_for_ts_release();

	fprintf(stdout, "Released!\n");

	//sleep_next_period();
	//CALL( task_mode(BACKGROUND_TASK) );

	return 0;
}

