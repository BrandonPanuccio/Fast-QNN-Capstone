import torch
from brevitas.nn import QuantLinear
from brevitas.quant import Int8ActPerTensorFloat

torch.manual_seed(0)


float_input = torch.randn(3, 2)
quant_linear = QuantLinear(2, 4, input_quant=Int8ActPerTensorFloat, bias=False)

quant_output = quant_linear(float_input)

print(f"Float input:\n {float_input} \n")
print(f"Quant output:\n {quant_output}")