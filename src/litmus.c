#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <sys/mman.h>

#include <sched.h> /* for cpu sets */

#include "litmus.h"
#include "internal.h"

void show_rt_param(struct rt_task* tp)
{
	printf("rt params:\n\t"
	       "exec_cost:\t%llu\n\tperiod:\t\t%llu\n\tcpu:\t%d\n",
	       tp->exec_cost, tp->period, tp->cpu);
}

task_class_t str2class(const char* str)
{
	if      (!strcmp(str, "hrt"))
		return RT_CLASS_HARD;
	else if (!strcmp(str, "srt"))
		return RT_CLASS_SOFT;
	else if (!strcmp(str, "be"))
		return RT_CLASS_BEST_EFFORT;
	else
		return -1;
}

#define NS_PER_MS 1000000

/* only for best-effort execution: migrate to target_cpu */
int be_migrate_to(int target_cpu)
{
	cpu_set_t cpu_set;

	CPU_ZERO(&cpu_set);
	CPU_SET(target_cpu, &cpu_set);
	return sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set);
}

int sporadic_task(lt_t e, lt_t p, lt_t phase,
		  int cpu, task_class_t cls,
		  budget_policy_t budget_policy, int set_cpu_set)
{
	return sporadic_task_ns(e * NS_PER_MS, p * NS_PER_MS, phase * NS_PER_MS,
				cpu, cls, budget_policy, set_cpu_set);
}

int sporadic_task_ns(lt_t e, lt_t p, lt_t phase,
			int cpu, task_class_t cls,
			budget_policy_t budget_policy, int set_cpu_set)
{
	struct rt_task param;
	int ret;

	/* Zero out first --- this is helpful when we add plugin-specific
	 * parameters during development.
	 */
	memset(&param, 0, sizeof(param));

	param.exec_cost = e;
	param.period    = p;
	param.cpu       = cpu;
	param.cls       = cls;
	param.phase	= phase;
	param.budget_policy = budget_policy;

	if (set_cpu_set) {
		ret = be_migrate_to(cpu);
		check("migrate to cpu");
	}
	return set_rt_task_param(gettid(), &param);
}

int init_kernel_iface(void);

int init_litmus(void)
{
	int ret, ret2;

	ret = mlockall(MCL_CURRENT | MCL_FUTURE);
	check("mlockall()");
	ret2 = init_rt_thread();
	return (ret == 0) && (ret2 == 0) ? 0 : -1;
}

int init_rt_thread(void)
{
	int ret;

        ret = init_kernel_iface();
	check("kernel <-> user space interface initialization");
	return ret;
}

void exit_litmus(void)
{
	/* nothing to do in current version */
}

int open_kfmlp_gpu_sem(int fd, int name, int num_gpus, int gpu_offset, int num_simult_users, int affinity_aware)
{
	int lock_od;
	int affinity_od;
	int num_replicas;
	struct gpu_affinity_observer_args aff_args;
	int aff_type;
	
	// number of GPU tokens
	num_replicas = num_gpus * num_simult_users;

	// create the GPU token lock
	lock_od = open_kfmlp_sem(fd, name, (void*)&num_replicas);
	if(lock_od < 0) {
		perror("open_kfmlp_sem");
		return -1;
	}
	
	// create the affinity method to use.
	// "no affinity" -> KFMLP_SIMPLE_GPU_AFF_OBS
	aff_args.obs.lock_od = lock_od;
	aff_args.replica_to_gpu_offset = gpu_offset;
	aff_args.nr_simult_users = num_simult_users;
	
	aff_type = (affinity_aware) ? KFMLP_GPU_AFF_OBS : KFMLP_SIMPLE_GPU_AFF_OBS;
	affinity_od = od_openx(fd, aff_type, name+1, &aff_args);
	if(affinity_od < 0) {
		perror("open_kfmlp_aff");
		return -1;
	}	
	
	return lock_od;
}



int open_ikglp_gpu_sem(int fd, int name, int num_gpus, int gpu_offset, int num_simult_users, int affinity_aware, int relax_max_fifo_len)
{
	int lock_od;
	int affinity_od;
	int num_replicas;
	struct gpu_affinity_observer_args aff_args;
	int aff_type;

	// number of GPU tokens
	num_replicas = num_gpus * num_simult_users;
	
	// create the GPU token lock
	lock_od = open_ikglp_sem(fd, name, (void*)&num_replicas);
	if(lock_od < 0) {
		perror("open_ikglp_sem");
		return -1;
	}
	
	// create the affinity method to use.
	// "no affinity" -> KFMLP_SIMPLE_GPU_AFF_OBS
	aff_args.obs.lock_od = lock_od;
	aff_args.replica_to_gpu_offset = gpu_offset;
	aff_args.nr_simult_users = num_simult_users;
	aff_args.relaxed_rules = (relax_max_fifo_len) ? 1 : 0;
	
	aff_type = (affinity_aware) ? IKGLP_GPU_AFF_OBS : IKGLP_SIMPLE_GPU_AFF_OBS;
	affinity_od = od_openx(fd, aff_type, name+1, &aff_args);
	if(affinity_od < 0) {
		perror("open_ikglp_aff");
		return -1;
	}	
	
	return lock_od;
}
