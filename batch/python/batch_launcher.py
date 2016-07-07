from datetime import date 
import os
import sys
import subprocess
import time
import glob


def job_version(directory):
    version_date = "v_1_"+str(date.today())
    if os.path.isdir(directory):
        dirs= [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory,f)) and f[:2]=='v_']
        version_max = 0
        for d in dirs:
            version = int(d.split("_")[1])
            if version > version_max: version_max = version
        version_date = "v_"+str(version_max+1)+"_"+str(date.today())
    return version_date

def wait_jobs(directory, wait=15):
    jobnames = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(directory+'/jobs/*.sub')]
    while True:
        time.sleep(wait)
        done = True
        for job in jobnames:
            if not os.path.exists(directory+'/{}.done'.format(job)): done = False
        if done: break


def prepare_jobs(working_dir, exe, pars, name='batch'):
    # Create working area
    if not os.path.isdir(working_dir): os.makedirs(working_dir)
    # Create one subdirectory for each version
    version = job_version(working_dir)
    os.makedirs(working_dir+'/'+version)
    os.makedirs(working_dir+'/'+version+'/jobs')
    os.makedirs(working_dir+'/'+version+'/logs')
    job_dir = working_dir+'/'+version+'/jobs'
    log_dir = working_dir+'/'+version+'/logs'
    # Create job files
    for i,par in enumerate(pars):
        par_list = ['--'+key+' '+value for key,value in par.items()]
        with open(job_dir+'/'+name+'_{}.sub'.format(i), 'w') as script:
            print >>script, '#! /bin/bash'
            print >>script, 'uname -a'
            print >>script, 'export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch/'
            print >>script, 'source $VO_CMS_SW_DIR/cmsset_default.sh'
            # Somehow the 'source setupenv' is not working properly on batch,
            # (related to python virtualenv??) so copy here what is done in setupenv
            print >>script, 'cd', os.environ['CMSSW_BASE']
            # cmsenv alias is not working on batch?
            print >>script, 'eval `scramv1 runtime -sh`'
            print >>script, 'cd', os.environ['L1TSTUDIES_BASE']
            print >>script, 'source ./env/bin/activate'
            print >>script, 'export PYTHONPATH=$PWD/env/lib/python2.7/site-packages/:$PYTHONPATH'
            print >>script, 'cd', working_dir+'/'+version
            print >>script, exe, ' '.join(par_list), '&>', log_dir+'/'+name+'_{}.log'.format(i)
            print >>script, 'touch', name+'_{}.done'.format(i)
    time.sleep(1)
    return working_dir+'/'+version



def launch_jobs(working_dir, pars, name='batch', queue='cms', proxy='~/.t3/proxy.cert'):
    print 'Sending {0} jobs on {1}'.format(len(pars), queue+'@llrt3')
    print '==============='
    for i,par in enumerate(pars):
        qsub_args = []
        qsub_args.append('-k')
        qsub_args.append('oe')
        qsub_args.append('-N')
        qsub_args.append(name+'_{}'.format(i))
        qsub_args.append('-q')
        qsub_args.append(queue+'@llrt3')
        qsub_args.append('-v')
        qsub_args.append('X509_USER_PROXY='+proxy)
        qsub_args.append('-V')
        qsub_args.append(working_dir+'/jobs/'+name+'_{}.sub'.format(i))
        command = ['qsub'] + qsub_args
        #print ' '.join(command)
        subprocess.call(command)
    print '==============='


def main(workingdir, exe, pars, name='batch', queue='cms', proxy='~/.t3/proxy.cert'):
    version_dir = prepare_jobs(working_dir=workingdir, exe=exe, pars=pars, name=name)
    launch_jobs(working_dir=version_dir, pars=pars, name=name, queue=queue, proxy=proxy)



if __name__=='__main__':
    import optparse
    import importlib
    usage = 'usage: python %prog [options]'
    parser = optparse.OptionParser(usage)
    parser.add_option('--workingdir', dest='working_dir', help='Working directory', default='batch/')
    parser.add_option('--exe', dest='executable', help='Executable', default='test.exe')
    parser.add_option('--pars', dest='parameter_file', help='Python file containing the list of parameters ', default='pars.py')
    parser.add_option('--name', dest='name', help='Name of the batch jobs', default='batch')
    parser.add_option('--queue', dest='queue', help='Batch queue', default='cms')
    parser.add_option('--proxy', dest='proxy', help='Grid user proxy', default='~/.t3/proxy.cert')
    (opt, args) = parser.parse_args()
    current_dir = os.getcwd();
    sys.path.append(current_dir)
    # Remove the extension of the python file before module loading
    if opt.parameter_file[-3:]=='.py': opt.parameter_file = opt.parameter_file[:-3]
    parameters = importlib.import_module(opt.parameter_file).parameters
    main(workingdir=opt.working_dir, exe=opt.executable, pars=parameters, name=opt.name, queue=opt.queue, proxy=opt.proxy)
