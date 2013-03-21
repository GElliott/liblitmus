#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>

#include <sched.h>

#include "litmus.h"
#include "internal.h"

static void tperrorx(char* msg)
{
	fprintf(stderr,
		"Task %d: %s: %m",
		gettid(), msg);
	exit(-1);
}

/* common launch routine */
int __launch_rt_task(rt_fn_t rt_prog, void *rt_arg, rt_setup_fn_t setup,
		     void* setup_arg)
{
	int ret;
	int rt_task = fork();

	if (rt_task == 0) {
		/* we are the real-time task
		 * launch task and die when it is done
		 */
		rt_task = gettid();
		ret = setup(rt_task, setup_arg);
		if (ret < 0)
			tperrorx("could not setup task parameters");
		ret = task_mode(LITMUS_RT_TASK);
		if (ret < 0)
			tperrorx("could not become real-time task");
		exit(rt_prog(rt_arg));
	}

	return rt_task;
}

int create_rt_task(rt_fn_t rt_prog, void *arg, struct rt_task* param)
{
	if (param->budget_policy == NO_ENFORCEMENT) {
		/* This is only safe if the task to be launched does not peg the CPU.
		 That is, it must block frequently for I/O or call sleep_next_period()
		 at the end of each job. Otherwise, the task may peg the CPU. */
		//printf("Warning: running budget enforcement used.\n");
	}

	return __launch_rt_task(rt_prog, arg, (rt_setup_fn_t) set_rt_task_param, param);
}


#define SCHED_NORMAL 0
#define SCHED_LITMUS 6

int task_mode(int mode)
{
	struct sched_param param;
	int me     = gettid();
	int policy = sched_getscheduler(gettid());
	int old_mode = policy == SCHED_LITMUS ? LITMUS_RT_TASK : BACKGROUND_TASK;

	param.sched_priority = 0;
	if (old_mode == LITMUS_RT_TASK && mode == BACKGROUND_TASK) {
		/* transition to normal task */
		return sched_setscheduler(me, SCHED_NORMAL, &param);
	} else if (old_mode == BACKGROUND_TASK && mode == LITMUS_RT_TASK) {
		/* transition to RT task */
		return sched_setscheduler(me, SCHED_LITMUS, &param);
	} else {
		errno = -EINVAL;
		return -1;
	}
}
