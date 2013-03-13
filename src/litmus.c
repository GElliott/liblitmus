#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>


#include <sched.h> /* for cpu sets */

#include "litmus.h"
#include "internal.h"

#define LP(name) {name ## _SEM, #name}

static struct {
	int id;
	const char* name;
} protocol[] = {
	LP(FMLP),
	LP(SRP),
	LP(MPCP),
	LP(MPCP_VS),
	{MPCP_VS_SEM, "MPCP-VS"},
	LP(DPCP),
	LP(PCP),

	{FIFO_MUTEX, "FIFO"},
	LP(IKGLP),
	LP(KFMLP),

	{IKGLP_SIMPLE_GPU_AFF_OBS, "IKGLP-GPU-SIMPLE"},
	{IKGLP_GPU_AFF_OBS, "IKGLP-GPU"},
	{KFMLP_SIMPLE_GPU_AFF_OBS, "KFMLP-GPU-SIMPLE"},
	{KFMLP_GPU_AFF_OBS, "KFMLP-GPU"},

	{PRIOQ_MUTEX, "PRIOQ"},
};

#define NUM_PROTOS (sizeof(protocol)/sizeof(protocol[0]))

int lock_protocol_for_name(const char* name)
{
	int i;

	for (i = 0; i < NUM_PROTOS; i++)
		if (strcmp(name, protocol[i].name) == 0)
			return protocol[i].id;

	return -1;
}

const char* name_for_lock_protocol(int id)
{
	int i;

	for (i = 0; i < NUM_PROTOS; i++)
		if (protocol[i].id == id)
			return protocol[i].name;

	return "<UNKNOWN>";
}

int litmus_open_lock(
	obj_type_t protocol,
	int lock_id,
	const char* namespace,
	void *config_param)
{
	int fd, od;

	fd = open(namespace, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
	if (fd < 0)
		return -1;
	od = od_openx(fd, protocol, lock_id, config_param);
	close(fd);
	return od;
}



void show_rt_param(struct rt_task* tp)
{
	printf("rt params:\n\t"
	       "exec_cost:\t%llu\n\tperiod:\t\t%llu\n\tcpu:\t%d\n",
	       tp->exec_cost, tp->period, tp->cpu);
}

void init_rt_task_param(struct rt_task* tp)
{
	/* Defaults:
	 *  - implicit deadline (t->relative_deadline == 0)
	 *  - phase = 0
	 *  - class = RT_CLASS_SOFT
	 *  - budget policy = NO_ENFORCEMENT
	 *  - fixed priority = LITMUS_LOWEST_PRIORITY
	 *  - release policy = SPORADIC
	 *  - cpu assignment = 0
	 *
	 * User must still set the following fields to non-zero values:
	 *  - tp->exec_cost
	 *  - tp->period
	 *
	 * User must set tp->cpu to the appropriate value for non-global
	 * schedulers. For clusters, set tp->cpu to the first CPU in the
	 * assigned cluster.
	 */

	memset(tp, 0, sizeof(*tp));

	tp->cls = RT_CLASS_SOFT;
	tp->priority = LITMUS_LOWEST_PRIORITY;
	tp->budget_policy = NO_ENFORCEMENT;
	tp->release_policy = SPORADIC;
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
		return (task_class_t)(-1);
}

#define NS_PER_MS 1000000

int sporadic_global(lt_t e_ns, lt_t p_ns)
{
	struct rt_task param;

	init_rt_task_param(&param);
	param.exec_cost = e_ns;
	param.period = p_ns;

	return set_rt_task_param(gettid(), &param);
}

int sporadic_partitioned(lt_t e_ns, lt_t p_ns, int partition)
{
	int ret;
	struct rt_task param;

	ret = be_migrate_to_partition(partition);
	check("be_migrate_to_partition()");
	if (ret != 0)
		return ret;

	init_rt_task_param(&param);
	param.exec_cost = e_ns;
	param.period = p_ns;
	param.cpu = partition_to_cpu(partition);

	return set_rt_task_param(gettid(), &param);
}

int sporadic_clustered(lt_t e_ns, lt_t p_ns, int cluster, int cluster_size)
{
	int ret;
	struct rt_task param;

	ret = be_migrate_to_cluster(cluster, cluster_size);
	check("be_migrate_to_cluster()");
	if (ret != 0)
		return ret;

	init_rt_task_param(&param);
	param.exec_cost = e_ns;
	param.period = p_ns;
	param.cpu = cluster_to_first_cpu(cluster, cluster_size);

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

int open_kfmlp_gpu_sem(int fd, int name,
	unsigned int num_gpus, unsigned int gpu_offset, unsigned int rho,
	int affinity_aware)
{
	int lock_od;
	int affinity_od;
	unsigned int num_replicas;
	struct gpu_affinity_observer_args aff_args;
	int aff_type;

	// number of GPU tokens
	num_replicas = num_gpus * rho;

	// create the GPU token lock
	lock_od = open_kfmlp_sem(fd, name, num_replicas);
	if(lock_od < 0) {
		perror("open_kfmlp_sem");
		return -1;
	}

	// create the affinity method to use.
	// "no affinity" -> KFMLP_SIMPLE_GPU_AFF_OBS
	aff_args.obs.lock_od = lock_od;
	aff_args.replica_to_gpu_offset = gpu_offset;
	aff_args.rho = rho;

	aff_type = (affinity_aware) ? KFMLP_GPU_AFF_OBS : KFMLP_SIMPLE_GPU_AFF_OBS;
	affinity_od = od_openx(fd, aff_type, name+1, &aff_args);
	if(affinity_od < 0) {
		perror("open_kfmlp_aff");
		return -1;
	}

	return lock_od;
}


//int open_ikglp_gpu_sem(int fd, int name, int num_gpus, int gpu_offset, int rho, int affinity_aware, int relax_max_fifo_len)
//{
//	int lock_od;
//	int affinity_od;
//	int num_replicas;
//	struct gpu_affinity_observer_args aff_args;
//	int aff_type;
//
//	// number of GPU tokens
//	num_replicas = num_gpus * num_simult_users;
//
//	// create the GPU token lock
//	lock_od = open_ikglp_sem(fd, name, (void*)&num_replicas);
//	if(lock_od < 0) {
//		perror("open_ikglp_sem");
//		return -1;
//	}
//
//	// create the affinity method to use.
//	// "no affinity" -> KFMLP_SIMPLE_GPU_AFF_OBS
//	aff_args.obs.lock_od = lock_od;
//	aff_args.replica_to_gpu_offset = gpu_offset;
//	aff_args.nr_simult_users = num_simult_users;
//	aff_args.relaxed_rules = (relax_max_fifo_len) ? 1 : 0;
//
//	aff_type = (affinity_aware) ? IKGLP_GPU_AFF_OBS : IKGLP_SIMPLE_GPU_AFF_OBS;
//	affinity_od = od_openx(fd, aff_type, name+1, &aff_args);
//	if(affinity_od < 0) {
//		perror("open_ikglp_aff");
//		return -1;
//	}
//
//	return lock_od;
//}




int open_ikglp_sem(int fd, int name, unsigned int nr_replicas)
{
	struct ikglp_args args = {
		.nr_replicas = nr_replicas,
		.max_in_fifos = IKGLP_M_IN_FIFOS,
		.max_fifo_len = IKGLP_OPTIMAL_FIFO_LEN};

	return od_openx(fd, IKGLP_SEM, name, &args);
}



int open_gpusync_token_lock(int fd, int name,
							unsigned int num_gpus, unsigned int gpu_offset,
							unsigned int rho, unsigned int max_in_fifos,
							unsigned int max_fifo_len,
							int enable_affinity_heuristics)
{
	int lock_od;
	int affinity_od;

	struct ikglp_args args = {
		.nr_replicas = num_gpus*rho,
		.max_in_fifos = max_in_fifos,
		.max_fifo_len = max_fifo_len,
	};
	struct gpu_affinity_observer_args aff_args;
	int aff_type;

	if (!num_gpus || !rho) {
		perror("open_gpusync_sem");
		return -1;
	}

	if ((max_in_fifos != IKGLP_UNLIMITED_IN_FIFOS) &&
		(max_fifo_len != IKGLP_UNLIMITED_FIFO_LEN) &&
		(max_in_fifos > args.nr_replicas * max_fifo_len)) {
		perror("open_gpusync_sem");
		return(-1);
	}

	lock_od = od_openx(fd, IKGLP_SEM, name, &args);
	if(lock_od < 0) {
		perror("open_gpusync_sem");
		return -1;
	}

	// create the affinity method to use.
	aff_args.obs.lock_od = lock_od;
	aff_args.replica_to_gpu_offset = gpu_offset;
	aff_args.rho = rho;
	aff_args.relaxed_rules = (max_fifo_len == IKGLP_UNLIMITED_FIFO_LEN) ? 1 : 0;

	aff_type = (enable_affinity_heuristics) ? IKGLP_GPU_AFF_OBS : IKGLP_SIMPLE_GPU_AFF_OBS;
	affinity_od = od_openx(fd, aff_type, name+1, &aff_args);
	if(affinity_od < 0) {
		perror("open_gpusync_affinity");
		return -1;
	}

	return lock_od;
}
