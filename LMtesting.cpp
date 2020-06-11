#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <dirent.h>

#include <typeinfo>

using namespace std;

int main(int argc, const char* argv[]) {
  
  // Create the device we pass around based on whether CUDA is available.
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
  }

  // if (argc != 2) {
  //   std::cerr << "usage: LMtesting <path-to-exported-script-module>\n";
  //   return -1;
  // }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::string file("/home/rakesh/rishabh_workspace/Garbage/kaldi/egs/MemoryLM/Grecipe/data/pytorch/rnnlm/newmodel.pt");
    //path = path + file;
    //std::cout << path;
    //std::string::iterator st = std::remove(path.begin(), path.end(), ' ');
    //path.erase(st, path.end());
    module = torch::jit::load(file);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
  std::cout << "Language Model\n\n";

  // Hyper parameters
  int64_t embed_size = 100;
  int64_t hidden_size = 100;
  int64_t num_layers = 2;
  int64_t num_samples = 1000;  // the number of words to be sampled
  int64_t batch_size = 20;
  int64_t sequence_length = 35;
  int64_t num_epochs = 2;
  double learning_rate = 20;

  

  //module->eval();

  
  torch::Tensor indata1 = torch::zeros({sequence_length,batch_size}, torch::kLong);


  vector<torch::jit::IValue> inputs;
  vector<float> data(sequence_length*batch_size, 1.0);
  torch::Tensor data_tensor = torch::from_blob(data.data(), {sequence_length,batch_size}).cuda();
  torch::Tensor h0 = torch::from_blob(std::vector<float>(num_layers* batch_size * hidden_size, 0.0).data(), {num_layers, batch_size, hidden_size}).cuda();
  //torch::Tensor c0 = torch::from_blob(std::vector<float>(num_layers* batch_size * hidden_size, 0.0).data(), {num_layers, batch_size, hidden_size});

  //torch::jit::IValue tuple = torch::ivalue::Tuple::create(h0);
  //torch::Tensor output = module.forward({data_tensor, h0}).toTensor().cuda();
  //auto accessor = output.accessor<float, 2>();
  //torch.tensor batch_size = 20;
 // torch::Tensor batchsize = torch::tensor({20});
  auto hidden_map = module.get_method("get_initial_state")({torch::tensor({1})});
  //auto d = module.get_method("ret")({torch::zeros({4,6}, torch::kLong)});
  //int i = d;

  torch::Tensor t = hidden_map.toTensor();
  //int val = d.toInt();
  //std::cout << t;
  //cout<<typeid(d).name();
  //torch::Tensor *t = new torch::Tensor(hidden_map);
  //std::cout << hidden_map;

  //auto bosword = torch::tensor({1.5}).cuda();
 //auto hidden2 = module.get_method("single_step_log")({hidden_map, bosword});
 //cout<<hidden2;
 //cout<<hidden;
  //cout << bosword;
  //std::cout << hidden1 ;


  std::vector<torch::Tensor*> state_to_context_;
  state_to_context_.push_back(new torch::Tensor(hidden_map.toTensor()));

  int word = 56;
  torch::Tensor thisword = torch::tensor({word}) ;
  torch::Tensor *new_context = new torch::Tensor();
  *new_context = thisword;
  std::cout<<*new_context;
} 
