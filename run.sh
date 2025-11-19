# For TCR-SREM + esm2_t6_8M_UR50D
#################################
python run.py \
    --config "configs/contactareareg/esm2_t6_8M_UR50D.json" \
    --batchsize "512" \
    --device "cuda:0" \
    --epoch "150" \
    --numworkers "8"

# For TCR-SREM + esm2_t12_35M_UR50D
###################################
# python run.py \
#     --config "configs/contactareareg/esm2_t12_35M_UR50D.json" \
#     --batchsize "512" \
#     --device "cuda:0" \
#     --epoch "150" \
#     --numworkers "8"

# For TCR-SREM + esm2_t33_650M_UR50D
###################################
# python run.py \
#     --config "configs/contactareareg/esm2_t33_650M_UR50D.json" \
#     --batchsize "512" \
#     --device "cuda:0" \
#     --epoch "150" \
#     --numworkers "8"

# For TCR-SREM + esm1b_t33_650M_UR50S
###################################
# python run.py \
#     --config "configs/contactareareg/esm1b_t33_650M_UR50S.json" \
#     --batchsize "512" \
#     --device "cuda:0" \
#     --epoch "150" \
#     --numworkers "8"

# For TCR-SREM + proteinbert
###################################
# python run.py \
#     --config "configs/contactareareg/proteinbert.json" \
#     --batchsize "512" \
#     --device "cuda:0" \
#     --epoch "150" \
#     --numworkers "8"