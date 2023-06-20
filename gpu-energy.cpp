#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include "rocm_smi/rocm_smi.h"

#define MAX_GPU 8

// g++ gpu-energy-amd.cpp -I/opt/rocm-5.2.3/include/ -L/opt/rocm-5.2.3/lib -o gpu-energy-amd -lrocm_smi64

using namespace std;

int printError(rsmi_status_t ret) {
  const char** statusString = NULL;
  rsmi_status_t r = rsmi_status_string(ret, statusString);

  if (r != RSMI_STATUS_SUCCESS || statusString == NULL)
    cerr << "RSMI Error (code): " << ret << endl;
  else
    cerr << "RSMI Error: " << *statusString << endl;
  return -1;
}


int main(int argc, char* argv[]) {
  rsmi_status_t ret;
  uint32_t numDevices;
  uint16_t devId;

  ret = rsmi_init(0);
  if (ret != RSMI_STATUS_SUCCESS)
    return printError(ret);

  ret = rsmi_num_monitor_devices(&numDevices);
  if (ret != RSMI_STATUS_SUCCESS)
    return printError(ret);

  if (numDevices < 1 || numDevices > MAX_GPU) {
    cerr << "ERROR: detected invalid number of GPU devices: " << numDevices << endl;
    return -1;
  }

  vector<float> prevValues(numDevices, -1);
  
  if (argc > 1) {
    string inFn = argv[1];
    cout << "Reading in previous energy values from " << inFn << endl;
    ifstream infile(inFn);

    unsigned int i;
    uint64_t power;
    float counterResolution;
    while (infile >> i >> power >> counterResolution) {
      if (i >= MAX_GPU) {
        cerr << "ERROR: invalid GPU index: " << i << " in " << inFn << endl;
        return -1;
      }
      
      prevValues[i] = power*counterResolution;
      cout << "READ: " << i << " " << power << " " << counterResolution << endl;
    }
    
  }
  
  string fn = "gpu-energy";
  if (const char* env_p = getenv("SLURM_JOB_ID"))
    fn = fn + "-" + env_p;
  
  filesystem::path ff(filesystem::temp_directory_path() / fn);

  ofstream fp(ff);
  if (!fp) {
    cerr << "ERROR: Unable to write to temporary file " << fn << endl;
    return -1;
  }    

  for (unsigned int i=0; i<numDevices; ++i) {
    uint64_t power;
    float counterResolution;
    uint64_t timestamp;
    ret = rsmi_dev_energy_count_get(i, &power, &counterResolution, &timestamp);
    if (ret != RSMI_STATUS_SUCCESS)
      return printError(ret);
    fp << i << " " << power << " " << counterResolution << endl;
  }

  fp.close();
  ret = rsmi_shut_down();

  cout << ff.c_str() << endl;
  return 0;
}
