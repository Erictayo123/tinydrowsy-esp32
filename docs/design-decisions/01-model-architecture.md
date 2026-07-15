# ```markdown

# \# Decision: Model Architecture

# 

# \*\*Date:\*\* 2026-07-15

# \*\*Status:\*\* Accepted

# 

# \## Context

# The goal was to create a lightweight CNN for eye-state classification that could run on an ESP32-S3 with limited memory and compute resources.

# 

# \## Decision

# \*\*TinyEyeNetV2\*\* - A depthwise-separable CNN with:

# \- 10,690 parameters

# \- 64×64 grayscale input

# \- 2-class output (open/closed)

# \- No softmax in model (applied in firmware)

# 

# \## Rationale

# 1\. \*\*Depthwise-separable convolutions\*\* reduce parameters by \~10× compared to standard convolutions

# 2\. \*\*64×64 input\*\* balances accuracy and compute

# 3\. \*\*No softmax\*\* allows direct logit comparison on embedded device

# 4\. \*\*10,690 parameters\*\* fits easily in 27.4 KB after INT8 quantization

# 

# \## Alternatives Considered

# | Model | Parameters | Acc. | Size | Verdict |

# |-------|-----------|------|------|---------|

# | MobileNetV2 | 2.2M | 94% | 14 MB | Too large |

# | ResNet-18 | 11M | 95% | 45 MB | Too large |

# | \*\*TinyEyeNetV2\*\* | \*\*10.6K\*\* | \*\*99.87%\*\* | \*\*27.4 KB\*\* | \*\*✅ Selected\*\* |

# 

# \## Consequences

# \- ✅ Fits in 27.4 KB after INT8 quantization

# \- ✅ 99.87% accuracy exceeds requirements

# \- ✅ Easy to deploy on ESP32-S3

# \- ❌ Requires software downscaling from 160×120 to 64×64

