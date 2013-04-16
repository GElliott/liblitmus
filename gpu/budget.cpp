#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
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

#define NUMS 4096
static int nums[NUMS];

inline static lt_t cputime_ns(void)
{
	struct timespec ts;
	lt_t time;
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);

	// safe, as long as sizeof(ls_t) >= 8
	time = s2ns(ts.tv_sec) + ts.tv_nsec;

	return time;
}

inline static lt_t wtime_ns(void)
{
	struct timespec ts;
	lt_t time;
	clock_gettime(CLOCK_MONOTONIC, &ts);

	// safe, as long as sizeof(ls_t) >= 8
	time = s2ns(ts.tv_sec) + ts.tv_nsec;

	return time;
}

static int loop_once(void)
{
	int i, j = 0;
	for (i = 0; i < NUMS; ++i)
		j += nums[i]++;
	return j;
}

int loop_for(lt_t time)
{
	lt_t end, now;
	lt_t last_loop = 0, loop_start;
	int dummy = 0;

	last_loop = 0;

	now = cputime_ns();
	end = now + time;

	/* '+ last_loop' attempts to avoid overrun */
	while (now + last_loop < end) {
		loop_start = now;
		dummy += loop_once();
		now = cputime_ns();
		last_loop = now - loop_start;
	}

	return dummy;
}

int OVERRUN = 0;
int SIGNALS = 0;
int BLOCK_SIGNALS_ON_SLEEP = 0;
int OVERRUN_RATE = 1; /* default: every job overruns */

int CXS_OVERRUN = 0;
int NUM_LOCKS = 1;
int NUM_REPLICAS = 1;
int NAMESPACE = 0;
int *LOCKS = NULL;
int IKGLP_LOCK = 0;
int USE_DGLS = 0;
int NEST_IN_IKGLP = 0;

int WAIT = 0;

enum eLockType
{
	FIFO,
	PRIOQ,
	IKGLP
};

eLockType LOCK_TYPE = FIFO;

int OVERRUN_BY_SLEEP = 0;

int NUM_JOBS = 0;
int NUM_COMPLETED_JOBS = 0;
int NUM_OVERRUNS = 0;

lt_t overrun_extra = 0;

int job(lt_t exec_ns, lt_t budget_ns)
{
	++NUM_JOBS;

	try{
		lt_t approx_remaining = budget_ns;
		lt_t now = cputime_ns();
		loop_for(lt_t(exec_ns * 0.9)); /* fudge it a bit to account for overheads */

		if (OVERRUN) {
			// do we want to overrun this job?
			if ((NUM_JOBS % OVERRUN_RATE) == 0) {
				approx_remaining -= (cputime_ns() - now);

				if (SIGNALS && BLOCK_SIGNALS_ON_SLEEP)
					block_litmus_signals(SIG_BUDGET);

				if(CXS_OVERRUN) {
					if (NEST_IN_IKGLP)
						litmus_lock(IKGLP_LOCK);
					if (USE_DGLS)
						litmus_dgl_lock(LOCKS, NUM_LOCKS);
					else
						for(int i = 0; i < NUM_LOCKS; ++i)
							litmus_lock(LOCKS[i]);
				}
				
				// intentionally overrun via suspension
				if (OVERRUN_BY_SLEEP)
					lt_sleep(approx_remaining + overrun_extra);
				else
					loop_for((approx_remaining + overrun_extra) * 0.9);

				if(CXS_OVERRUN) {
					if (USE_DGLS)
						litmus_dgl_unlock(LOCKS, NUM_LOCKS);
					else
						for(int i = NUM_LOCKS-1; i >= 0; --i)
							litmus_unlock(LOCKS[i]);						
					if (NEST_IN_IKGLP)
						litmus_unlock(IKGLP_LOCK);
				}
				
				if (SIGNALS && BLOCK_SIGNALS_ON_SLEEP)
					unblock_litmus_signals(SIG_BUDGET);
			}
		}
		++NUM_COMPLETED_JOBS;
	}
	catch (const litmus::sigbudget& e) {
		++NUM_OVERRUNS;
	}

	sleep_next_period();
	return 1;
}

#define OPTSTR "SbosOvzalwqixdn:r:"

int main(int argc, char** argv)
{
	int ret;

	srand(getpid());

	lt_t e_ns = ms2ns(2);
	lt_t p_ns = ms2ns(50) + rand()%200;
	lt_t budget_ns = p_ns/2;
	lt_t duration = s2ns(60);
	lt_t terminate_time;
	unsigned int first_job, last_job;
	int opt;
	struct rt_task param;
	budget_drain_policy_t drain_policy = DRAIN_SIMPLE;
	int compute_overrun_rate = 0;
	int once = 1;


	while ((opt = getopt(argc, argv, OPTSTR)) != -1) {
		switch(opt) {
		case 'S':
			SIGNALS = 1;
			break;
		case 'b':
			BLOCK_SIGNALS_ON_SLEEP = 1;
			break;
		case 's':
			OVERRUN_BY_SLEEP = 1;
			break;
		case 'o':
			OVERRUN = 1;
			overrun_extra = budget_ns/2;
			break;
		case 'O':
			OVERRUN = 1;
			overrun_extra = 4*p_ns;
			break;
		case 'a':
			/* select an overrun rate such that a task should be caught
			 * up from a backlog caused by an overrun before the next
			 * overrun occurs.
			 */
			compute_overrun_rate = 1;
			break;
		case 'v':
			drain_policy = DRAIN_SOBLIV;
			break;
		case 'z':
			drain_policy = DRAIN_SIMPLE_IO;
			break;
		case 'l':
			CXS_OVERRUN = 1;
			NAMESPACE = open("semaphores", O_RDONLY | O_CREAT, S_IRUSR | S_IWUSR);
			break;
		case 'q':
			LOCK_TYPE = PRIOQ;
			break;
		case 'i':
			LOCK_TYPE = IKGLP;
			break;
		case 'x':
			NEST_IN_IKGLP = 1;
			break;
		case 'w':
			WAIT = 1;
			break;
		case 'd':
			USE_DGLS = 1;
			break;
		case 'n':
			NUM_LOCKS = atoi(optarg);
			break;
		case 'r':
			NUM_REPLICAS = atoi(optarg);
			break;
		case ':':
			printf("missing argument\n");
			assert(false);
			break;
		default:
			printf("unknown option\n");
			assert(false);
			break;
		}
	}

	assert(!BLOCK_SIGNALS_ON_SLEEP || (BLOCK_SIGNALS_ON_SLEEP && SIGNALS));
	assert(!CXS_OVERRUN || (CXS_OVERRUN && WAIT));
	assert(LOCK_TYPE != IKGLP || NUM_LOCKS == 1);
	assert(LOCK_TYPE != IKGLP || (LOCK_TYPE == IKGLP && !NEST_IN_IKGLP));
	assert(NUM_LOCKS > 0);
	if (LOCK_TYPE == IKGLP || NEST_IN_IKGLP)
		assert(NUM_REPLICAS >= 1);
	
	LOCKS = new int[NUM_LOCKS];

	if (compute_overrun_rate) {
		int backlog = (int)ceil((overrun_extra + budget_ns)/(double)budget_ns);
		if (!CXS_OVERRUN)
			OVERRUN_RATE = backlog + 2; /* some padding */
		else
			OVERRUN_RATE = 2*backlog + 2; /* overrun less frequently for testing */
	}

	init_rt_task_param(&param);
	param.exec_cost = budget_ns;
	param.period = p_ns;
	param.release_policy = PERIODIC;
	param.drain_policy = drain_policy;
	if (!SIGNALS)
		param.budget_policy = PRECISE_ENFORCEMENT;
	else
		param.budget_signal_policy = PRECISE_SIGNALS;

	init_litmus();

	ret = set_rt_task_param(gettid(), &param);
	assert(ret == 0);

	if (CXS_OVERRUN) {
		int i;
		for(i = 0; i < NUM_LOCKS; ++i) {
			int lock = -1;
			switch(LOCK_TYPE)
			{
				case FIFO:
					lock = open_fifo_sem(NAMESPACE, i);
					break;
				case PRIOQ:
					lock = open_prioq_sem(NAMESPACE, i);
					break;
				case IKGLP:
					lock = open_ikglp_sem(NAMESPACE, i, NUM_REPLICAS);
					break;
			}
			if (lock < 0) {
				perror("open_sem");
				exit(-1);
			}
			LOCKS[i] = lock;
		}
		
		if (NEST_IN_IKGLP) {
			IKGLP_LOCK = open_ikglp_sem(NAMESPACE, i, NUM_REPLICAS);
			if (IKGLP_LOCK < 0) {
				perror("open_sem");
				exit(-1);
			}
		}
	}
	
	if (WAIT) {
		ret = wait_for_ts_release();
		if (ret < 0)
			perror("wait_for_ts_release");
	}
	
	ret = task_mode(LITMUS_RT_TASK);
	assert(ret == 0);

	sleep_next_period();

	ret = get_job_no(&first_job);
	assert(ret == 0);

	terminate_time = duration + wtime_ns();

	while (wtime_ns() < terminate_time) {
		try{
			if(once) {
				activate_litmus_signals(SIG_BUDGET, litmus::throw_on_litmus_signal);
				once = 0;
			}
			job(e_ns, budget_ns);
		}
		catch(const litmus::sigbudget &e) {
			/* drop silently */
		}
	}

	ret = get_job_no(&last_job);
	assert(ret == 0);

	ret = task_mode(BACKGROUND_TASK);
	assert(ret == 0);

	printf("# Kernel Jobs: %d\n", last_job - first_job + 1);
	printf("# User Started Jobs: %d\n", NUM_JOBS);
	printf("# User Jobs Completed: %d\n", NUM_COMPLETED_JOBS);
	printf("# Overruns: %d\n", NUM_OVERRUNS);

	delete[] LOCKS;
	
	return 0;
}
