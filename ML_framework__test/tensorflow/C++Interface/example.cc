#include <iostream>
#include <vector>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

int main() {
  /*Scope contains common name space properties while adding ops*/
  Scope root_scope = Scope::NewRootScope();
  auto A = Const(root_scope, { {1.3f, 2.8f, 4.5f},
                              {6.f, 3.9f, 11.2f},
                              {-3.4f, 4.6f, 1.9f} } );
  auto b = Const(root_scope, { {1.f, 2.f, 3.f} });
  auto v = MatMul(root_scope.WithOpName("mat_vec_product"), A, b, MatMul::TransposeB(true));
  vector<Tensor> outputs;
  ClientSession session(root_scope);
  session.Run({v}, &outputs);
  cout << "Output of tensor computation: " << endl;
  for(int i=0;i<3;i++){
    cout <<  outputs[i].matrix<float>() << " ";
  }
  return 0;
}
