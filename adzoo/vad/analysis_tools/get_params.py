import jittor

YOUR_CKPT_PATH = None
file_path = YOUR_CKPT_PATH
model = jittor.load(file_path)
all = 0
for key in list(model['state_dict'].keys()):
    all += model['state_dict'][key].nelement()
print(all)
