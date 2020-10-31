import sys
sys.path.insert(0,'./src/')

import sys
import os
import subprocess
import argparse
import time
import threading
import datetime
import dateutil.relativedelta
import copy
import random
import utils.utils as utils
import json
import pdb
import shlex

def filter_lines(joblines):
    new_joblines = []
    for jobline in joblines:
        if jobline == '':
            continue
        if jobline.lstrip().startswith('#'):
            continue
        if jobline.lstrip().startswith('python'):
            new_joblines.append(jobline)
        if jobline.lstrip().startswith('bash'):
            new_joblines.append(jobline)
        if jobline.lstrip().startswith('export'):
            new_joblines.append(jobline)
        
    return new_joblines

def get_jobline(args, lock):
    lock.acquire()
    with open(args.jobs_file, "r") as f:
        lines = f.readlines()
        
    lines = filter_lines(lines)
    
    n_jobs = len(lines)
    if n_jobs == 0:
        lock.release()
        return None, n_jobs

    jobline = lines[0]
    del lines[0]

    with open(args.jobs_file, "w") as f:
        for line in lines:
            f.write(line)
            
    lock.release()
                
    return jobline.lstrip().rstrip(), n_jobs


def add_to_file(filename, lock, jobline):
    lock.acquire()
    with open(filename, "a+") as f:
        f.write(jobline)
    lock.release()

    
def remove_from_running(args, lock, threadid):
    lock.acquire()
    with open(args.running_jobs_file, "r") as f:
        lines = f.readlines()
    
    with open(args.running_jobs_file, "w") as f:
        for l in range(len(lines)-1, -1, -1):
            if threadid == int(lines[l].split(' ')[0]):
                del lines[l]
            else:
                f.write(lines[l])
    lock.release()

    
def is_cuda_out_of_memory(output_file):
    with open(output_file, "r") as f:
        lines = f.readlines()
    
    for line in reversed(lines):
        line_l = line.lower()
        
        if ('out of memory' in line_l and 'cuda' in line_l) or \
            'CUDA-capable devices are busy or unavailable' in line_l:
            return True
        
    return False

                
def get_available_gpus(gpu_file):
    with open(gpu_file, 'r') as f:
        gpus_available = [int(g) for g in f.readline().split(',')]
    return gpus_available


def now(date_format=None):
    if date_format=="full":
        return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    else:
        return datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')

    
def idle_thread(gpu, threadid, mult):
    time.sleep(mult)
    

def thread_main(args, gpu, lock, threadid, total_jobs):
    
    while True:
        while gpu not in get_available_gpus(args.gpu_file):
            mult = random.randint(30, 90)
            idle_thread(gpu, threadid, mult)
            if not args.idle_finished:
                with open(args.jobs_file, "r") as f:
                    lines = f.readlines()
                if len(lines) == 0:
                    break
        
        jobline, jobs_left = get_jobline(args, lock)
        
        if jobline is None:
            if not args.idle_finished:
                print("{} Thread {} (GPU {}) has finished.".format(now(), threadid, gpu))
                break
            else:
                mult = random.randint(30, 90)
                idle_thread(gpu, threadid, mult)
                continue
            
        if jobline == '':
            continue
            
        if jobline.startswith('#'):
            continue
        
        preamble = None
        if jobline.startswith('@'):
            preamble = jobline.split(" ")[0]
            allocated_gpu = preamble[1]
            
            if int(allocated_gpu) != int(gpu):
                print("Skipping", jobline)
                add_to_file(args.jobs_file, lock, jobline + '\n')
                time.sleep(1.0)
                continue
            
            jobline_cmd = ' '.join(jobline.split(" ")[1:])
        else:
            jobline_cmd = jobline
           
        if "&>" in jobline_cmd:
            job = jobline_cmd.split("&>")[0].lstrip().rstrip()
            output_file = jobline_cmd.split("&>")[1].lstrip().rstrip()
        else:
            job = jobline_cmd
            output_file = "output.txt"
            
        my_cmd = shlex.split(job.format(**dict(gpu=gpu)))
        if args.dummy_run: my_cmd.append("--dummy_run")
        
        add_to_file(args.running_jobs_file, lock, "{} {}\n".format(threadid, jobline_cmd))
        
        if my_cmd[0] == "python":
            args_file = jobline_cmd.split("--args_file ")[1].split(" ")[0]
        elif my_cmd[0] == "bash":
            with open(my_cmd[1], "r") as f:
                script_content = f.read()
            args_file = script_content.split("--args_file ")[1].split(" ")[0]
            
#         with open(args_file) as f:
#             args_file = json.load(fp=f)
#             expname = args_file['experiment_name']
            
        start = time.time()
        print("{}\t{} jobs left. Thread {} (GPU {}) running {}".format(now(), jobs_left-1, threadid, gpu, args_file))
        with open(output_file, "w") as f:
            process = subprocess.run(my_cmd, stdout=f, stderr=f)
        end = time.time()
        
        if process.returncode == 0:
            start = datetime.datetime.fromtimestamp(start)
            end = datetime.datetime.fromtimestamp(end)
            diff = dateutil.relativedelta.relativedelta(end, start)

            diff= '{}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}'.format(diff.years, diff.months, diff.days, 
                                                                 diff.hours, diff.minutes, diff.seconds)
            end=end.strftime('%Y-%m-%d %H:%M:%S')
            start=start.strftime('%Y-%m-%d %H:%M:%S')

            line = "START:{},END:{},DUR:{},CMD:{}\n".format(start, end, diff, jobline_cmd)
            add_to_file(args.completed_jobs_file, lock, line)
            remove_from_running(args, lock, threadid)
            
        else:
            # if out of memory error add job back to file and wait a few mins
            if is_cuda_out_of_memory(output_file):
                add_to_file(args.jobs_file, lock, jobline + '\n')
                remove_from_running(args, lock, threadid)
                mult = random.randint(15, 30)*WAIT_FACTOR
                print("{} Thread {} (GPU {}) out of GPU memory. Going to sleep for {} mins".format(
                    now(), threadid, gpu, int(mult/60)))
                idle_thread(gpu, threadid, mult)
            else:
                print("{} Failed job! See: {}".format(now(), output_file))
                add_to_file(args.failed_jobs_file, lock, jobline + '\n')
                remove_from_running(args, lock, threadid)

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs='+', default=[0])
    parser.add_argument("--jobs_per_gpu", type=int, default=4)
    parser.add_argument("--gpu_file", type=str, default="job_utils/gpus.txt")
    parser.add_argument("--jobs_file", type=str, default="job_utils/new.txt")
    parser.add_argument("--completed_jobs_file", type=str, default="job_utils/completed.txt")
    parser.add_argument("--running_jobs_file", type=str, default="job_utils/running.txt")
    parser.add_argument("--failed_jobs_file", type=str, default="job_utils/failed.txt")
    parser.add_argument("--dummy_run", type=utils.str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--idle_finished", type=utils.str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--mute", type=int, nargs='+', default=[])  # temporarely stop running gpus
    args = parser.parse_args()
    
    if args.dummy_run:
        print("Dummy Run")
        WAIT_FACTOR=2
    else:
        WAIT_FACTOR=60
        
    gpus_available = args.gpus
    if len(gpus_available) == 0:
        gpus_available = get_available_gpus(args.gpu_file)
    else:
        # save given gpus, mute those that are out
        temp = [gpu for gpu in gpus_available if gpu not in mute]
        with open(args.gpu_file, "w") as f:
            f.write(','.join(map(str,temp)))
    
    if len(gpus_available) == 0 or args.jobs_per_gpu <= 0:
        print("more than one gpu required")
        sys.exit(1)
        
    if not os.path.isfile(os.path.abspath(args.jobs_file)):
        print("job file does not exist {}".format(os.path.abspath(args.jobs_file)))
        sys.exit(1)
    
    print("Sorting out {} file".format(args.jobs_file))
    with open(args.jobs_file, "r") as f:
        lines = f.readlines()
              
    lines = filter_lines(lines)
              
    with open(args.jobs_file, "w") as f:
        for l in lines:
            f.write(l)
              
    num_jobs = len(lines)
    print("{}\t{} jobs found!".format(now('full'),num_jobs))
    
    if not args.idle_finished and num_jobs == 0:
        sys.exit(0)
    
    lock=threading.Lock()
    num_workers = args.jobs_per_gpu * len(gpus_available)
    
    for w in range(num_workers):
        gpu = gpus_available[w % len(gpus_available)]
        t = threading.Thread(target=thread_main, args=(args,gpu,lock,w,num_jobs))
        t.start()
        time.sleep(1)
        
    main_thread = threading.currentThread()
    for t in threading.enumerate():
        if t is not main_thread:
            t.join()
            time.sleep(1)
    print("{} Jobs done!".format(now("full")))
    