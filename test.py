import torch

a = torch.randn(3, 3, 2)
print(f'a:{a}')
print(f'a.shape:{a.shape}')

'''
b = torch.unsqueeze(a, 0)
print(f'b:{b}')
print(f'b.shape:{b.shape}')

c = torch.unsqueeze(a, 1)
print(f'c:{c}')
print(f'c.shape:{c.shape}')

d = torch.unsqueeze(a, 2)
print(f'd:{d}')
print(f'd.shape:{d.shape}')
'''