#ifndef LITMUS_H
#define LITMUS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/types.h>
#include <stdint.h>
#include <setjmp.h>

/* Include kernel header.
 * This is required for the rt_param
 * and control_page structures.
 */
#include "litmus/rt_param.h"
#include "litmus/signal.h"

#include "asm/cycles.h" /* for null_call() */

typedef int pid_t;	 /* PID of a task */

/* obtain the PID of a thread */
pid_t gettid(void);

/* migrate to partition */
int be_migrate_to(int target_cpu);

int set_rt_task_param(pid_t pid, struct rt_task* param);
int get_rt_task_param(pid_t pid, struct rt_task* param);

/* setup helper */

/* Times are given in ms. The 'priority' parameter
 * is only relevant under fixed-priority scheduling (and
 * ignored by other plugins). The task_class_t parameter
 * is ignored by most plugins.
 */
int sporadic_task(
		lt_t e, lt_t p, lt_t phase,
		int partition, unsigned int priority,
		task_class_t cls,
		budget_policy_t budget_policy,
		budget_signal_policy_t budget_signal_policy,
		int set_cpu_set);

/* Times are given in ns. The 'priority' parameter
 * is only relevant under fixed-priority scheduling (and
 * ignored by other plugins). The task_class_t parameter
 * is ignored by most plugins.
 */
int sporadic_task_ns(
		lt_t e, lt_t p, lt_t phase,
		int cpu, unsigned int priority,
		task_class_t cls,
		budget_policy_t budget_policy,
		budget_signal_policy_t budget_signal_policy,
		int set_cpu_set);

/* Convenience macros. Budget enforcement off by default in these macros. */
#define sporadic_global(e, p) \
	sporadic_task(e, p, 0, 0, LITMUS_LOWEST_PRIORITY, \
		RT_CLASS_SOFT, NO_ENFORCEMENT, NO_SIGNALS, 0)
#define sporadic_partitioned(e, p, cpu) \
	sporadic_task(e, p, 0, cpu, LITMUS_LOWEST_PRIORITY, \
		RT_CLASS_SOFT, NO_ENFORCEMENT, NO_SIGNALS, 1)

/* file descriptor attached shared objects support */
typedef enum  {
	FMLP_SEM	= 0,
	SRP_SEM		= 1,
	MPCP_SEM	= 2,
	MPCP_VS_SEM	= 3,
	DPCP_SEM	= 4,
	PCP_SEM		= 5,

	RSM_MUTEX	= 6,
	IKGLP_SEM	= 7,
	KFMLP_SEM	= 8,
	
	IKGLP_SIMPLE_GPU_AFF_OBS = 9,
	IKGLP_GPU_AFF_OBS = 10,
	KFMLP_SIMPLE_GPU_AFF_OBS = 11,
	KFMLP_GPU_AFF_OBS = 12,
} obj_type_t;

int lock_protocol_for_name(const char* name);
const char* name_for_lock_protocol(int id);

int od_openx(int fd, obj_type_t type, int obj_id, void* config);
int od_close(int od);

static inline int od_open(int fd, obj_type_t type, int obj_id)
{
	return od_openx(fd, type, obj_id, 0);
}

/* real-time locking protocol support */
int litmus_lock(int od);
int litmus_unlock(int od);

/* Dynamic group lock support.  ods arrays MUST BE PARTIALLY ORDERED!!!!!!
 * Use the same ordering for lock and unlock.
 *
 * Ex:
 *   litmus_dgl_lock({A, B, C, D}, 4);
 *   litmus_dgl_unlock({A, B, C, D}, 4);
 */
int litmus_dgl_lock(int* ods, int dgl_size);
int litmus_dgl_unlock(int* ods, int dgl_size);	

/* nvidia graphics cards */
int register_nv_device(int nv_device_id);
int unregister_nv_device(int nv_device_id);

/* job control*/
int get_job_no(unsigned int* job_no);
int wait_for_job_release(unsigned int job_no);
int sleep_next_period(void);

/*  library functions */
int  init_litmus(void);
int  init_rt_thread(void);
void exit_litmus(void);

/* A real-time program. */
typedef int (*rt_fn_t)(void*);

/* These two functions configure the RT task to use enforced exe budgets */
int create_rt_task(rt_fn_t rt_prog, void *arg, int cpu, int wcet, int period);
int __create_rt_task(rt_fn_t rt_prog, void *arg, int cpu, int wcet,
		     int period, task_class_t cls);

/*	per-task modes */
enum rt_task_mode_t {
	BACKGROUND_TASK = 0,
	LITMUS_RT_TASK  = 1
};
int task_mode(int target_mode);

void show_rt_param(struct rt_task* tp);
task_class_t str2class(const char* str);

/* non-preemptive section support */
void enter_np(void);
void exit_np(void);
int  requested_to_preempt(void);

/* task system support */
int wait_for_ts_release(void);
int release_ts(lt_t *delay);
int get_nr_ts_release_waiters(void);


int enable_aux_rt_tasks(int flags);
int disable_aux_rt_tasks(int flags);

#define __NS_PER_MS 1000000

static inline lt_t ms2lt(unsigned long milliseconds)
{
	return __NS_PER_MS * milliseconds;
}

/* sleep for some number of nanoseconds */
int lt_sleep(lt_t timeout);

/* CPU time consumed so far in seconds */
double cputime(void);

/* wall-clock time in seconds */
double wctime(void);

/* semaphore allocation */

static inline int open_fmlp_sem(int fd, int name)
{
	return od_open(fd, FMLP_SEM, name);
}

static inline int open_kfmlp_sem(int fd, int name, void* arg)
{
	return od_openx(fd, KFMLP_SEM, name, arg);
}

static inline int open_srp_sem(int fd, int name)
{
	return od_open(fd, SRP_SEM, name);
}

static inline int open_pcp_sem(int fd, int name, int cpu)
{
	return od_openx(fd, PCP_SEM, name, &cpu);
}

static inline int open_mpcp_sem(int fd, int name)
{
	return od_open(fd, MPCP_SEM, name);
}

static inline int open_dpcp_sem(int fd, int name, int cpu)
{
	return od_openx(fd, DPCP_SEM, name, &cpu);
}

static inline int open_rsm_sem(int fd, int name)
{
	return od_open(fd, RSM_MUTEX, name);
}

static inline int open_ikglp_sem(int fd, int name, void *arg)
{
	return od_openx(fd, IKGLP_SEM, name, arg);
}
	
static inline int open_kfmlp_simple_gpu_aff_obs(int fd, int name,
	struct gpu_affinity_observer_args *arg)
{
	return od_openx(fd, KFMLP_SIMPLE_GPU_AFF_OBS, name, arg);
}
	
static inline int open_kfmlp_gpu_aff_obs(int fd, int name,
	struct gpu_affinity_observer_args *arg)
{
	return od_openx(fd, KFMLP_GPU_AFF_OBS, name, arg);
}

static inline int open_ikglp_simple_gpu_aff_obs(int fd, int name, void *arg)
{
	return od_openx(fd, IKGLP_SIMPLE_GPU_AFF_OBS, name, arg);
}	
	
static inline int open_ikglp_gpu_aff_obs(int fd, int name, void *arg)
{
	return od_openx(fd, IKGLP_GPU_AFF_OBS, name, arg);
}

// takes names "name" and "name+1"
int open_kfmlp_gpu_sem(int fd, int name, int num_gpus, int gpu_offset,
		int num_simult_users, int affinity_aware);
int open_ikglp_gpu_sem(int fd, int name, int num_gpus, int gpu_offset,
		int num_simult_users, int affinity_aware, int relax_max_fifo_len);
	
/* syscall overhead measuring */
int null_call(cycles_t *timestamp);

/*
 * get control page:
 * atm it is used only by preemption migration overhead code
 * but it is very general and can be used for different purposes
 */
struct control_page* get_ctrl_page(void);


/* sched_trace injection */
int inject_name(void);
int inject_param(void); /* sporadic_task_ns*() must have already been called */
int inject_release(lt_t release, lt_t deadline, unsigned int job_no);
int inject_completion(unsigned int job_no);


/* Litmus signal handling */

typedef struct litmus_sigjmp
{
	sigjmp_buf env;
	struct litmus_sigjmp *prev;
} litmus_sigjmp_t;

void push_sigjmp(litmus_sigjmp_t* buf);
litmus_sigjmp_t* pop_sigjmp(void);

typedef void (*litmus_sig_handler_t)(int);
typedef void (*litmus_sig_actions_t)(int, siginfo_t *, void *);

/* ignore specified signals. all signals raised while ignored are dropped */
void ignore_litmus_signals(unsigned long litmus_sig_mask);

/* register a handler for the given set of litmus signals */
void activate_litmus_signals(unsigned long litmus_sig_mask,
				litmus_sig_handler_t handler);

/* register an action signal handler for a given set of signals */
void activate_litmus_signal_actions(unsigned long litmus_sig_mask,
				litmus_sig_actions_t handler);

/* Block a given set of litmus signals. Any signals raised while blocked
 * are queued and delivered after unblocking. Call ignore_litmus_signals()
 * before unblocking if you wish to discard these. Blocking may be
 * useful to protect COTS code in Litmus that may not be able to deal
 * with exception-raising signals.
 */
void block_litmus_signals(unsigned long litmus_sig_mask);

/* Unblock a given set of litmus signals. */
void unblock_litmus_signals(unsigned long litmus_sig_mask);

#define SIG_BUDGET_MASK			0x00000001
/* more ... */

#define ALL_LITMUS_SIG_MASKS	(SIG_BUDGET_MASK)

/* Try/Catch structures useful for implementing abortable jobs.
 * Should only be used in legitimate cases. ;)
 */
#define LITMUS_TRY \
do { \
	int sigsetjmp_ret_##__FUNCTION__##__LINE__; \
	litmus_sigjmp_t lit_env_##__FUNCTION__##__LINE__; \
	push_sigjmp(&lit_env_##__FUNCTION__##__LINE__); \
	sigsetjmp_ret_##__FUNCTION__##__LINE__ = \
		sigsetjmp(lit_env_##__FUNCTION__##__LINE__.env, 1); \
	if (sigsetjmp_ret_##__FUNCTION__##__LINE__ == 0) {

#define LITMUS_CATCH(x) \
	} else if (sigsetjmp_ret_##__FUNCTION__##__LINE__ == (x)) {

#define END_LITMUS_TRY \
	} /* end if-else-if chain */ \
} while(0); /* end do from 'LITMUS_TRY' */

/* Calls siglongjmp(signum). Use with TRY/CATCH.
 * Example:
 *  activate_litmus_signals(SIG_BUDGET_MASK, longjmp_on_litmus_signal);
 */
void longjmp_on_litmus_signal(int signum);

#ifdef __cplusplus
}
#endif




#ifdef __cplusplus
/* Expose litmus exceptions if C++.
 *
 * KLUDGE: We define everything in the header since liblitmus is a C-only
 * library, but this header could be included in C++ code.
 */

#include <exception>

namespace litmus
{
	class litmus_exception: public std::exception
	{
	public:
		litmus_exception() throw() {}
		virtual ~litmus_exception() throw() {}
		virtual const char* what() const throw() { return "litmus_exception";}
	};

	class sigbudget: public litmus_exception
	{
	public:
		sigbudget() throw() {}
		virtual ~sigbudget() throw() {}
		virtual const char* what() const throw() { return "sigbudget"; }
	};

	/* Must compile your program with "non-call-exception". */
	static __attribute__((used))
	void throw_on_litmus_signal(int signum)
	{
		/* We have to unblock the received signal to get more in the future
		 * because we are not calling siglongjmp(), which normally restores
		 * the mask for us.
		 */
		if (SIG_BUDGET == signum) {
			unblock_litmus_signals(SIG_BUDGET_MASK);
			throw sigbudget();
		}
		/* else if (...) */
		else {
			/* silently ignore */
		}
	}

}; /* end namespace 'litmus' */

#endif /* end __cplusplus */

#endif
