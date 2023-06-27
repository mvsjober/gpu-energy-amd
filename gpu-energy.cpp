#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#include "rocm_smi/rocm_smi.h"

#define EPSILON 0.0001f

using namespace std;

struct energy_counter_t {
    uint64_t energy;
    float resolution;
    uint64_t timestamp;
};

map<unsigned int, energy_counter_t> previousValues;


// Generate "unique" temp filename based on Slurm variables
string generateFilename() {
  string fn = "gpu-energy";

  if (const char* job_id = getenv("SLURM_JOB_ID"))
    fn += string("-") + job_id;

  if (const char* proc_id = getenv("SLURM_PROCID"))
    fn += string("-") + proc_id;
    
  return filesystem::path(filesystem::temp_directory_path() / fn);
}


int main(int argc, char* argv[]) {
  bool save = false;   // if we should save the result to file
  bool diff = false;  // if we should print energy difference

  
  // Handle command line arguments
  string mode = argc > 1 ? argv[1] : "";
  string filename = argc > 2 ? argv[2] : "";

  if (argc <= 3 && (mode == "--save" || mode == "-s"))
    save = true;
  else if (argc <= 3 && (mode == "--diff" || mode == "-p"))
    diff = true;
  else if (!mode.empty()) {
    cerr << "Usage: " << argv[0] << endl;
    cerr << "       " << argv[0] << " --save [filename]" << endl;
    cerr << "       " << argv[0] << " --diff [filename]" << endl;
    return -1;
  }
    

  // Initialize RSMI library
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


  // Resolve filename if needed
  if ((save || diff) && filename.empty()) {
    filename = generateFilename();
  }
  
  
  // Read previous values if needed
  if (diff) {
    ifstream fp(filename);

    if (!fp) {
      cerr << "ERROR: unable to read from file " << filename << endl;
      return -1;
    }
    
    unsigned int i;
    while (fp)  {
      energy_counter_t cnt;
      fp >> i >> cnt.energy >> cnt.resolution >> cnt.timestamp;
      previousValues[i] = cnt;
    }
    
    if (i+1 != numDevices) {
      cerr << "ERROR: number of devices read from " << filename
           << " does not match: " << i+1 << " != " << numDevices << endl;
      return -1;
    }
  }

  
  // Open file for writing
  ofstream fp;
  if (save) {
    if (filesystem::exists(filename)) {
      cerr << "WARNING: temporary file " << filename << " already exists!"
           << endl;
    }
    fp.open(filename);
    if (!fp) {
      cerr << "ERROR: unable to write to file " << filename << endl;
      return -1;
    }
  }    


  // Loop over all visible GPU devices and print/save energy counter
  for (unsigned int i=0; i<numDevices; ++i) {
    // uint64_t energy;
    // float res;
    // uint64_t timestamp;
    energy_counter_t cnt;
    ret = rsmi_dev_energy_count_get(i, &cnt.energy, &cnt.resolution,
                                    &cnt.timestamp);
    
    if (ret != RSMI_STATUS_SUCCESS) {
      cerr << "ERROR: unable to get GPU energy counter from ROCm library"
           << endl;
      return -1;
    }

    if (save) {
      fp << i << " " << cnt.energy << " " << cnt.resolution << " "
         << cnt.timestamp << endl;
    }

    uint64_t energy = cnt.energy;

    // Check previous value if we're printing the difference
    if (diff) {
      auto prev = previousValues.at(i);
      if (fabs(cnt.resolution - prev.resolution) > EPSILON) {
        cerr << "ERROR: counter resolutions are different: " << cnt.resolution
             << " != " << prev.resolution << endl;
        return -1;
      }
      if (prev.energy > energy) {
        cerr << "ERROR: previous energy counter larger than current value: "
             << prev.energy << " > " << energy << endl;
        return -1;
      }
      energy -= prev.energy;
    }

    if (diff || !save) {
      double energyWh = (double)energy*cnt.resolution/3600000000.0;
      cout << "GPU " << i << ": " << energyWh << " Wh" << endl;
    }      
  }

  if (save)
    fp.close();
  if (diff)
    filesystem::remove(filename);
    
  ret = rsmi_shut_down();

  return 0;
}
