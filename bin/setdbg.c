#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "litmus.h"

int main(int argc, char** argv)
{
	pid_t target;
	int flag;
	int ret = -1;

	if(argc != 3)
	{
		fprintf(stderr, "Not enough arguments.\n");
		goto out;
	}

	target = atoi(argv[1]);
	flag = atoi(argv[2]);

	if(target <= 0)
	{
		fprintf(stderr, "Bad pid: %d\n", target);
		goto out;
	}

	ret = set_dbg(target, flag);
	if(ret != 0)
	{
		fprintf(stderr, "Bad return value: %d\n", ret);
	}

out:
	return ret;
}
