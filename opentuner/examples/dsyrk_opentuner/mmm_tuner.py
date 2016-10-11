#!/usr/bin/env python
#
# Optimize blocksize of apps/mmm_block.c
#
# This is an extremely simplified version meant only for tutorials
#
import adddeps  # fix sys.path

import opentuner
from opentuner import ConfigurationManipulator
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result


class GccFlagsTuner(MeasurementInterface):

  def manipulator(self):
    """
    Define the search space by creating a
    ConfigurationManipulator
    """
    manipulator = ConfigurationManipulator()
    manipulator.add_parameter(
      IntegerParameter('TILE_SIZE', 2, 256))
    return manipulator

  def run(self, desired_result, input, limit):
    """
    Compile and run a given configuration then
    return performance
    """
    cfg = desired_result.configuration.data

    CFLAGS =  '-fopenmp -fPIC -O3 -std=c99 -Wall -pedantic -Wshadow -Wno-unused '
    CFLAGS += '-DPLASMA_WITH_MKL -DMKL_Complex16="double _Complex" -DMKL_Complex8="float _Complex" '
    INC    =  '-I/home/pedro/plasma_autotuner/include '
    INC    += '-I/home/pedro/plasma_autotuner/test '
    INC    += '-I/opt/intel/compilers_and_libraries_2016.3.210/linux/mkl/include '
    LIBS   =  '-L/opt/intel/compilers_and_libraries_2016.3.210/linux/mkl/lib -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lm '
    LIBS   += '-L/home/pedro/plasma_autotuner/lib -lplasma -lcoreblas '

    gcc_cmd = 'gcc  -c  {0}  {1}  {2}  -D{3}={4}  dccrb2cm.c    -o  dccrb2cm.o && '.format(INC,CFLAGS,LIBS,'TILE_SIZE',cfg['TILE_SIZE']) 
    gcc_cmd += 'gcc  -c  {0}  {1}  {2}  -D{3}={4}  dcm2ccrb.c    -o  dcm2ccrb.o && '.format(INC,CFLAGS,LIBS,'TILE_SIZE',cfg['TILE_SIZE']) 
    gcc_cmd += 'gcc  -c  {0}  {1}  {2}  -D{3}={4}  core_dgemm.c  -o  core_dgemm.o && '.format(INC,CFLAGS,LIBS,'TILE_SIZE',cfg['TILE_SIZE'])  
    gcc_cmd += 'gcc  -c  {0}  {1}  {2}  -D{3}={4}  pdgemm.c      -o  pdgemm.o && '.format(INC,CFLAGS,LIBS,'TILE_SIZE',cfg['TILE_SIZE'])  
    gcc_cmd += 'gcc  -c  {0}  {1}  {2}  -D{3}={4}  dgemm.c       -o  dgemm.o && '.format(INC,CFLAGS,LIBS,'TILE_SIZE',cfg['TILE_SIZE'])  
    gcc_cmd += 'gcc  -c  {0}  {1}  {2}  -D{3}={4}  core_dsyrk.c  -o  core_dsyrk.o && '.format(INC,CFLAGS,LIBS,'TILE_SIZE',cfg['TILE_SIZE'])  
    gcc_cmd += 'gcc  -c  {0}  {1}  {2}  -D{3}={4}  pdsyrk.c      -o  pdsyrk.o && '.format(INC,CFLAGS,LIBS,'TILE_SIZE',cfg['TILE_SIZE'])  
    gcc_cmd += 'gcc  -c  {0}  {1}  {2}  -D{3}={4}  dsyrk.c       -o  dsyrk.o && '.format(INC,CFLAGS,LIBS,'TILE_SIZE',cfg['TILE_SIZE'])  
    gcc_cmd += 'gcc  dcm2ccrb.o dccrb2cm.o core_dgemm.o pdgemm.o dgemm.o core_dsyrk.o pdsyrk.o dsyrk.o {0}  {1}  {2}  -D{3}={4}  test_opentuner.c  -o  ./tmp.bin'.format(INC,CFLAGS,LIBS,'TILE_SIZE',cfg['TILE_SIZE'])  

    #gcc_cmd = 'gcc -c -fopenmp mmm_block.cpp'  
    #gcc_cmd += ' -D{0}={1}'.format('TILE_SIZE',cfg['TILE_SIZE'])
    #gcc_cmd += ' -o ./tmp.bin'

    #print(gcc_cmd)

    compile_result = self.call_program(gcc_cmd)
    assert compile_result['returncode'] == 0

    run_cmd = './tmp.bin'

    run_result = self.call_program(run_cmd)
    assert run_result['returncode'] == 0

    return Result(time=run_result['time'])

  def save_final_config(self, configuration):
    """called at the end of tuning"""
    print "Optimal block size written to mmm_final_config.json:", configuration.data
    self.manipulator().save_to_file(configuration.data,
                                    'mmm_final_config.json')


if __name__ == '__main__':
  argparser = opentuner.default_argparser()
  GccFlagsTuner.main(argparser.parse_args())
