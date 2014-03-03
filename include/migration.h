
typedef int pid_t;

/* obtain the PID of a thread */
pid_t gettid();

/* Assign a task to a cpu.
 * PRECOND: tid is not yet in real-time mode (it's a best effort task).
 * Set tid == 0 to migrate the caller */
int be_migrate_thread_to_cpu(pid_t tid, int target_cpu);

/* Assign a task to a scheduling domain (cluster, partition, etc.)
 * PRECOND: (1) tid is not yet in real-time mode.
 *          (2) plugin that supports /proc/litmus/domain is active.
 */
int be_migrate_thread_to_cluster(pid_t tid, int domain);

int be_migrate_to_cpu(int target_cpu);
int be_migrate_to_domain(int domain);


int num_online_cpus();
int release_master();
int domain_to_cpus(int domain, unsigned long long int* mask);
int cpu_to_domains(int cpu, unsigned long long int* mask);

int domain_to_first_cpu(int domain);


/* deprecated functions. */

/* Calls be_migrate_to_domain(), dropping cluster_size. */
int be_migrate_to_cluster(int cluster, int cluster_size);
/* Calls domain_to_first_cpu(), dropping cluster_size. */
int cluster_to_first_cpu(int cluster, int cluster_size);
/* Calls domain_to_first_cpu(). */
int partition_to_cpu(int partition);
