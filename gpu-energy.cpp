#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include "rocm_smi/rocm_smi.h"

#define EPSILON 0.0001f

using namespace std;

int main(int argc, char* argv[]) {
  bool doDiff = false;
  rsmi_status_t ret;
  uint32_t numDevices = 0;

  ret = rsmi_init(0);
  if (ret != RSMI_STATUS_SUCCESS) {
    cerr << "ERROR: unable to initialize ROCm SMI library" << endl;
    return -1;
  }

  ret = rsmi_num_monitor_devices(&numDevices);
  if (ret != RSMI_STATUS_SUCCESS || numDevices < 1) {
    cerr << "ERROR: unable to detect available ROCm devices" << endl;
    return -1;
  }

  vector<uint64_t> prevValues(numDevices, -1);
  float prevResolution = -1.0;

  if (argc == (int)numDevices + 2) {
    doDiff = true;
    prevResolution = atof(argv[1]);
  
    for (int i=2; i<argc; ++i) {
      prevValues[i-2] = atol(argv[i]);
    }
  } else if (argc != 1) {
    cerr << "Usage: " << argv[0] << endl;
    cerr << "       " << argv[0] << " <counterResolution> <energy1> ... <energy"
         << numDevices << ">" << endl;
    return -1;
  }
  
  float counterResolution = prevResolution;
  ostringstream output;

  for (unsigned int i=0; i<numDevices; ++i) {
    uint64_t energy;
    float res;
    uint64_t timestamp;
    ret = rsmi_dev_energy_count_get(i, &energy, &res, &timestamp);
    if (counterResolution < 0.0) {
      counterResolution = res;
      output << counterResolution << " ";
    } else if (fabs(res - counterResolution) > EPSILON) {
      cerr << "ERROR: counter resolution is not constant: " << res << " != "
           << counterResolution << endl;
      return -1;
    }
    counterResolution = res;
    
    if (ret != RSMI_STATUS_SUCCESS) {
      cerr << "ERROR: unable to get GPU energy counter from ROCm library" << endl;
      return -1;
    }

    if (doDiff) {
      uint64_t prevEnergy = prevValues[i];
      if (prevEnergy < 0.0) {
        cerr << "Previous value for GPU " << i << " incorrect " << prevEnergy
             << endl;
        return -1;
      }
      energy = energy - prevEnergy;
    }

    if (doDiff) {
      double energyWh = energy*(double)counterResolution/3600000000.0;
      cout << "GPU " << i << ": " << energyWh << " Wh" << endl;
    } else  {
      output << energy << " ";
    }
  }
  if (!doDiff)
    cout << output.str() << endl;

  ret = rsmi_shut_down();

  return 0;
}
