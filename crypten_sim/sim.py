import torch
import torch.nn.functional as F

# Number of neurons for each relu, grouped by the search granularity
relu_costs = {
    #"resnet18": [(32,), (8,), (8,), (8,), (8,), (4,), (4,), (4,), (4,), (2,), (2,), (2,), (2,), (1,), (1,), (1,), (1,)],
    #"resnet18": [(32,), (8, 8), (8, 8), (4, 4), (4, 4), (2, 2), (2, 2), (1, 1), (1, 1)],
    "resnet18": [(32,), (8, 8, 8, 8), (4, 4, 4, 4), (2, 2, 2, 2), (1, 1, 1, 1)],
    "resnet34": [(32,), (8, 8) * 3, (4, 4) * 4, (2, 2) * 6, (1, 1) * 3],
    "resnet50": [(32,), (8, 8, 32) * 3, (16, 4, 16) + (4, 4, 16) * 3, (8, 2, 8) + (2, 2, 8) * 5, (4, 1, 4) + (1, 1, 4) * 2],
    "vgg11_bn": [(32,), (16,), (8, 8), (4, 4), (1, 1), (2, 2)],
    "vgg16_bn": [(32, 32), (16, 16), (8, 8, 8), (4, 4, 4), (1, 1, 1), (2, 2)],
    }

def encode(x):
    # This is from crypten's encoder.encode() and
    # common.generate_random_ring_element()
    x = (x * 65536).long()
    #TODO: TMP: Why does this overflow?
    #ring_size = 18446744073709551616
    ring_size = 2 ** 64
    rand = torch.randint(
            -(ring_size // 2),
            (ring_size - 1) // 2,
            x.shape,
            #generator=torch.Generator(device=torch.device("cpu")),
            dtype=torch.long,
            device=x.device
        )
    x0 = x + rand
    x1 = -rand
    return x0, x1

def decode(x0, x1):
    # This is from crypten's encoder.decode().
    tensor = x0 + x1
    correction = (tensor < 0).long()
    dividend = tensor.div(65536 - correction, rounding_mode="floor")
    remainder = tensor % 65536
    remainder += (remainder == 0).long() * 65536 * correction
    tensor = dividend.float() + remainder.float() / 65536
    return tensor

class CryptenSim():
    def __init__(self):
        self.relu_idx = 0
        self.relu_compression_params = {}
        self.is_sim = False
        self.model_name = ""

    def init_simulator(self, model_name):
        self.model_name = model_name
        self.is_sim = True

    def init_relu_idx(self):
        if self.is_sim:
            self.relu_idx = 0

    def simulated_relu(self, x):
        if not self.is_sim:
            return F.relu(x)
        if self.relu_idx not in self.relu_compression_params:
            self.relu_compression_params[self.relu_idx] = (64, 0)
        N, M = self.relu_compression_params[self.relu_idx]
        self.relu_idx += 1
        # Using bits in (N, M]
        # Total N - M bits
        x0, x1 = encode(x)
        def truncate(x, n, m):
            if n == 64:
                return (x >> m)
            else:
                return (x >> m).bitwise_and((2 ** (n-m)) - 1)
        if N - M == 0:
            return x
        x0 = truncate(x0, N, M)
        x1 = truncate(x1, N, M)
        return x * (((x0 + x1) >> (N - M - 1)).bitwise_and(1) == 0)

    def set_relu_params(self, level, msb, lsb):
        start = 0
        for i, group in enumerate(relu_costs[self.model_name]):
            if i == level:
                for j in range(len(group)):
                    self.relu_compression_params[start + j] = (msb, lsb)
                break
            else:
                start += len(group)

    def get_relu_costs(self):
        return relu_costs[self.model_name]

simulator = CryptenSim()

