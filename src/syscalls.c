/* To get syscall() we need to define _GNU_SOURCE
 * in modern glibc versions.
 */

/* imported from the kernel source tree */
#include "asm/unistd.h"

/* for syscall() */
#include <unistd.h>

#include "litmus.h"

/*	Syscall stub for setting RT mode and scheduling options */

pid_t gettid(void)
{
	return syscall(__NR_gettid);
}

int set_rt_task_param(pid_t pid, struct rt_task *param)
{
	if (param->budget_signal_policy != NO_SIGNALS) {
		/* drop all signals until they're explicitly activated by
		 * user code. */
		ignore_litmus_signals(SIG_BUDGET);
	}

	return syscall(__NR_set_rt_task_param, pid, param);
}

int get_rt_task_param(pid_t pid, struct rt_task *param)
{
	return syscall(__NR_get_rt_task_param, pid, param);
}

int sleep_next_period(void)
{
	return syscall(__NR_complete_job);
}

int od_openx(int fd, obj_type_t type, int obj_id, void *config)
{
	return syscall(__NR_od_open, fd, type, obj_id, config);
}

int od_close(int od)
{
	return syscall(__NR_od_close, od);
}

int litmus_lock(int od)
{
	return syscall(__NR_litmus_lock, od);
}

int litmus_unlock(int od)
{
	return syscall(__NR_litmus_unlock, od);
}

int litmus_should_yield_lock(int od)
{
	return syscall(__NR_litmus_should_yield_lock, od);
}

int litmus_dgl_lock(int *ods, int dgl_size)
{
	return syscall(__NR_litmus_dgl_lock, ods, dgl_size);
}

int litmus_dgl_unlock(int *ods, int dgl_size)
{
	return syscall(__NR_litmus_dgl_unlock, ods, dgl_size);
}

int litmus_dgl_should_yield_lock(int *ods, int dgl_size)
{
	return syscall(__NR_litmus_dgl_should_yield_lock, ods, dgl_size);
}

int get_job_no(unsigned int *job_no)
{
	return syscall(__NR_query_job_no, job_no);
}

int wait_for_job_release(unsigned int job_no)
{
	return syscall(__NR_wait_for_job_release, job_no);
}

int sched_setscheduler(pid_t pid, int policy, int* priority)
{
	return syscall(__NR_sched_setscheduler, pid, policy, priority);
}

int sched_getscheduler(pid_t pid)
{
	return syscall(__NR_sched_getscheduler, pid);
}

static int __wait_for_ts_release(struct timespec *release)
{
	return syscall(__NR_wait_for_ts_release, release);
}

int wait_for_ts_release(void)
{
	return __wait_for_ts_release(NULL);
}

int wait_for_ts_release2(struct timespec *release)
{
	return __wait_for_ts_release(release);
}

int release_ts(lt_t *delay)
{
	return syscall(__NR_release_ts, delay);
}

int null_call(cycles_t *timestamp)
{
	return syscall(__NR_null_call, timestamp);
}

int enable_aux_rt_tasks(int flags)
{
	return syscall(__NR_set_aux_tasks, flags | AUX_ENABLE);
}

int disable_aux_rt_tasks(int flags)
{
	return syscall(__NR_set_aux_tasks, flags & ~AUX_ENABLE);
}

int inject_name(void)
{
	return syscall(__NR_sched_trace_event, ST_INJECT_NAME, NULL);
}

int inject_param(void)
{
	return syscall(__NR_sched_trace_event, ST_INJECT_PARAM, NULL);
}

int inject_release(lt_t release, lt_t deadline, unsigned int job_no)
{
	struct st_inject_args args = {.release = release, .deadline = deadline, .job_no = job_no};
	return syscall(__NR_sched_trace_event, ST_INJECT_RELEASE, &args);
}

int inject_completion(unsigned int job_no)
{
	struct st_inject_args args = {.release = 0, .deadline = 0, .job_no = job_no};
	return syscall(__NR_sched_trace_event, ST_INJECT_COMPLETION, &args);
}

int inject_gpu_migration(unsigned int to, unsigned int from)
{
	struct st_inject_args args = {.to = to, .from = from};
	return syscall(__NR_sched_trace_event, ST_INJECT_MIGRATION, &args);
}

int __inject_action(unsigned int action)
{
	struct st_inject_args args = {.action = action};
	return syscall(__NR_sched_trace_event, ST_INJECT_ACTION, &args);
}
