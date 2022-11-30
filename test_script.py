import os

checkpoint_dir = "./logs/run_01/111/policies/checkpoints/"
policy_dir = "/home/singamse/Mohan/circuit-training/logs/run_01/111/policies"

lst = os.listdir(checkpoint_dir)
lst.sort()

print("Latest checkpoint saved:", lst[-1])
x = os.path.join(checkpoint_dir, lst[-1])
print(x)