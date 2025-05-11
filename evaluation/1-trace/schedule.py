import asyncio
import math
import json  # 导入 json 模块，用于格式化输出
from uu import decode
from pathlib import Path

# 1) 在程序启动时读取外部 events.json
events_file = Path(__file__).parent /trace/"test2000s.json"
with open(events_file, "r") as f:
    events1 = json.load(f)

# 模拟 llm.engine 接口（实际使用时请替换为真实接口）
class Engine:
    async def remove(self, gpu_id):
        print(f"await llm.engine.remove({gpu_id})")

    async def add(self, actor, gpu_id):
        print(f"await llm.engine.add({actor},{gpu_id})")

    async def change_engine(self, original_gpu, original_role, new_gpu, new_role):
        # 参数格式：原来的 gpu id、原来的角色、新的 gpu id（同一值）和新的角色
        print(f"await llm.engine.change_engine({original_gpu}, {original_role!r}, {new_gpu}, {new_role!r})")


class DummyLLM:
    def __init__(self):
        self.engine = Engine()


llm = DummyLLM()


class GPUResourceManager:
    def __init__(self, n, a, start_time=0):
        self.n = n
        self.a = a
        self.prefillremove=0
        self.decoderemove =0
        # 初始化每个 GPU 的累计生存时间，单位自定义，初始均为 0
        self.gpu_lifetime = {i: 0 for i in range(n)}
        self.gpu_role = {}
        self.prefill_pool = []  # 存储当前分配到 prefill 池的 GPU id
        self.decode_pool = []  # 存储当前分配到 decode 池的 GPU id
        self.active_gpus = set(range(n))  # 初始所有 GPU 均活跃

        init_prefill = round((self.a / (self.a + 1)) * n)
        for gpu in range(n):
            if gpu < init_prefill:
                self.prefill_pool.append(gpu)
                self.gpu_role[gpu] = "prefill"
            else:
                self.decode_pool.append(gpu)
                self.gpu_role[gpu] = "decode"

        self.last_update_time = start_time
        self.interface_ops = []

    async def adjust_pools_2(self):
        # 当前池中的数量
        p = len(self.prefill_pool)
        d = len(self.decode_pool)
        # 复活待用 GPU 列表：不考虑其原有 gpu_role，无论如何都用 recovered 列表恢复
        recovered = [gpu for gpu in self.active_gpus
                     if gpu not in self.prefill_pool and gpu not in self.decode_pool
                     and self.gpu_lifetime[gpu] == 0]
        # 当前 active 数
        T = p + d + len(recovered)
        if T == 0 or self.a <= 0:
            return

        # 计算当前流水线吞吐量
        current_tp = min(d, p / self.a)
        # 初始化可转移数
        pt = 0
        dt = 0
        if current_tp > 0:
            if current_tp == d:
                p_min = math.ceil(self.a * d)
                pt = max(p - p_min, 0)
                dt = 0
            else:
                d_min = math.ceil(p / self.a)
                dt = max(d - d_min, 0)
                pt = 0
        else:
            pt = p
            dt = d

        if T < 2:
            return

        # 计算最优分配：遍历 decode 数量（1 ≤ d < T）选出能使流水线吞吐量最大的方案
        best_tp = -1
        best_d = None
        for d in range(1, T):
            current_tp = min(d, (T - d) / self.a)
            if current_tp > best_tp:
                best_tp = current_tp
                best_d = d
        d_max = best_d
        p_max = T - best_d

        # ===== 根据可转移情况调整 =====
        # 规则1：如果 pt == 0 且 dt == 0，则仅用 recovered GPU 补足各池到最优
        if pt == 0 and dt == 0 and current_tp != 0:
            while len(self.prefill_pool) < p_max and recovered:
                candidate = recovered.pop(0)
                self.prefill_pool.append(candidate)
                # 无论原角色如何，直接通过 add 加入 prefill 池
                self.interface_ops.append(("add", "prefill", candidate))
                self.gpu_role[candidate] = "prefill"
            while len(self.decode_pool) < d_max and recovered:
                candidate = recovered.pop(0)
                self.decode_pool.append(candidate)
                self.interface_ops.append(("add", "decode", candidate))
                self.gpu_role[candidate] = "decode"

        # 规则2：如果 pt > 0 且 dt == 0（当前瓶颈为 decode）
        elif pt > 0 and dt == 0 and current_tp != 0:
            if p < p_max:
                while len(self.prefill_pool) < p_max and recovered:
                    candidate = recovered.pop(0)
                    self.prefill_pool.append(candidate)
                    self.interface_ops.append(("add", "prefill", candidate))
                    self.gpu_role[candidate] = "prefill"
                while len(self.decode_pool) < d_max and recovered:
                    candidate = recovered.pop(0)
                    self.decode_pool.append(candidate)
                    self.interface_ops.append(("add", "decode", candidate))
                    self.gpu_role[candidate] = "decode"
            elif p == p_max:
                while len(self.decode_pool) < d_max and recovered:
                    candidate = recovered.pop(0)
                    self.decode_pool.append(candidate)
                    self.interface_ops.append(("add", "decode", candidate))
                    self.gpu_role[candidate] = "decode"
            else:
                surplus = p - p_max
                transfer_count = 0
                while surplus > 0 and transfer_count < surplus:
                    # 转移 prefill 池中“最年轻”的 GPU到 decode 池
                    candidate = min(self.prefill_pool, key=lambda gpu: self.gpu_lifetime[gpu])
                    self.prefill_pool.remove(candidate)
                    self.decode_pool.append(candidate)
                    self.interface_ops.append(("change_engine", candidate, "prefill", candidate, "decode"))
                    self.gpu_role[candidate] = "decode"
                    transfer_count += 1

        # 规则3：如果 dt > 0 且 pt == 0（当前瓶颈为 prefill）
        elif dt > 0 and pt == 0 and current_tp != 0:
            transfer_count = 0
            # 从 decode 池中转移“最年长”的 GPU到 prefill 池
            while len(self.prefill_pool) < p_max and transfer_count < dt and self.decode_pool:
                candidate = max(self.decode_pool, key=lambda gpu: self.gpu_lifetime[gpu])
                self.decode_pool.remove(candidate)
                self.prefill_pool.append(candidate)
                self.interface_ops.append(("change_engine", candidate, "decode", candidate, "prefill"))
                self.gpu_role[candidate] = "prefill"
                transfer_count += 1
            while len(self.prefill_pool) < p_max and recovered:
                candidate = recovered.pop(0)
                self.prefill_pool.append(candidate)
                self.interface_ops.append(("add", "prefill", candidate))
                self.gpu_role[candidate] = "prefill"
            while len(self.decode_pool) < d_max and recovered:
                candidate = recovered.pop(0)
                self.decode_pool.append(candidate)
                self.interface_ops.append(("add", "decode", candidate))
                self.gpu_role[candidate] = "decode"

        # 规则4：如果当前吞吐量为 0，则所有 active GPU 均可调整
        else:
            print("maxxxxxxxxxxxxxxxxxxx")
            p_max = math.ceil((self.a / (self.a + 1)) * T)
            d_max = T - p_max
            sorted_gpus = sorted(list(self.active_gpus), key=lambda gpu: self.gpu_lifetime[gpu])
            # 生存时间较长的分配给 prefill，较短的分配给 decode
            prefill_candidates = sorted_gpus[-p_max:]
            decode_candidates = sorted_gpus[:d_max]
            for gpu in prefill_candidates:
                if gpu in self.decode_pool:
                    self.decode_pool.remove(gpu)
                if gpu not in self.prefill_pool:
                    self.prefill_pool.append(gpu)
                # 即使原角色不同，也直接走 add 操作加入 prefill
                self.interface_ops.append(("add", "prefill", gpu))
                self.gpu_role[gpu] = "prefill"
            for gpu in decode_candidates:
                if gpu in self.prefill_pool:
                    self.prefill_pool.remove(gpu)
                if gpu not in self.decode_pool:
                    self.decode_pool.append(gpu)
                self.interface_ops.append(("add", "decode", gpu))
                self.gpu_role[gpu] = "decode"

    async def adjust_pools(self):
        """
        修改后的调整逻辑：
          1. 根据当前活跃 GPU 数 T 和参数 a，计算最优分配：
             - target_decode = d*，target_prefill = T - d*
          2. 调整 prefill_pool：
             - 如果不足，则优先使用恢复 GPU直接 add 进入 prefill_pool；
             - 如果恢复 GPU不足，则从 decode_pool中转换 GPU到 prefill_pool。
             - 如果过多，则直接从 prefill_pool转换 GPU到 decode_pool。
          3. 调整 decode_pool：
             - 如果不足，则优先使用恢复 GPU直接 add 进入 decode_pool；
             - 如果恢复 GPU不足，则从 prefill_pool转换 GPU到 decode_pool。
          4. 将剩余的恢复 GPU全部直接加入 decode_pool。
          在所有场景下，不再使用原有角色信息，所有恢复 GPU都直接用 add 操作，
          而转换操作也只考虑当前池的数量，不依赖原角色。
        """
        T = len(self.active_gpus)
        if T < 2:
            return

        # 计算最优分配
        best_tp = -1
        best_d = None
        for d in range(1, T):
            current_tp = min(d, (T - d) / self.a)
            if current_tp > best_tp:
                best_tp = current_tp
                best_d = d
        target_decode = best_d
        target_prefill = T - best_d

        # 找出所有恢复 GPU（生存时间==0 且未分配到任何池中）
        recovered = [
            gpu for gpu in self.active_gpus
            if gpu not in self.prefill_pool and gpu not in self.decode_pool and self.gpu_lifetime[gpu] == 0
        ]

        # 调整 prefill_pool 至目标 target_prefill
        while len(self.prefill_pool) < target_prefill:
            if recovered:
                candidate = recovered.pop(0)
                self.prefill_pool.append(candidate)
                self.interface_ops.append(("add", "prefill", candidate))
                self.gpu_role[candidate] = "prefill"
            else:
                # 恢复 GPU不足，从 decode_pool转换 GPU到 prefill_pool
                if self.decode_pool:
                    candidate = self.decode_pool.pop(0)
                    self.prefill_pool.append(candidate)
                    self.interface_ops.append(("change_engine", candidate, "decode", candidate, "prefill"))
                    self.gpu_role[candidate] = "prefill"
                else:
                    break

        while len(self.prefill_pool) > target_prefill:
            candidate = self.prefill_pool.pop()
            self.decode_pool.append(candidate)
            self.interface_ops.append(("change_engine", candidate, "prefill", candidate, "decode"))
            self.gpu_role[candidate] = "decode"

        # 调整 decode_pool 至目标 target_decode
        while len(self.decode_pool) < target_decode:
            if recovered:
                candidate = recovered.pop(0)
                self.decode_pool.append(candidate)
                self.interface_ops.append(("add", "decode", candidate))
                self.gpu_role[candidate] = "decode"
            else:
                if self.prefill_pool:
                    candidate = self.prefill_pool.pop(0)
                    self.decode_pool.append(candidate)
                    self.interface_ops.append(("change_engine", candidate, "prefill", candidate, "decode"))
                    self.gpu_role[candidate] = "decode"
                else:
                    break

        # 将所有剩余恢复 GPU全部加入 decode_pool
        for candidate in recovered:
            self.decode_pool.append(candidate)
            self.interface_ops.append(("add", "decode", candidate))
            self.gpu_role[candidate] = "decode"

    async def process_event(self, event):
        current_time = event.get("time", self.last_update_time)
        delta = current_time - self.last_update_time

        # 更新所有活跃 GPU（生存时间>=0）的累计生存时间
        for gpu in self.active_gpus:
            if self.gpu_lifetime[gpu] >= 0:
                self.gpu_lifetime[gpu] += delta
        self.last_update_time = current_time

        # --- Step 1：处理抢占 ---
        for gpu in event.get("preempted", []):
            if gpu in self.active_gpus:
                self.gpu_lifetime[gpu] = -1
                self.active_gpus.remove(gpu)
                if gpu in self.prefill_pool:
                    self.prefill_pool.remove(gpu)
                    self.prefillremove+=1
                if gpu in self.decode_pool:
                    self.decode_pool.remove(gpu)
                    self.decoderemove += 1
                self.interface_ops.append(("remove", gpu))

        # --- Step 2：处理恢复 ---
        for gpu in event.get("resumed", []):
            if gpu not in self.active_gpus:
                self.active_gpus.add(gpu)
                self.gpu_lifetime[gpu] = 0

        # --- Step 3：调整池分配以最大化流水线吞吐 ---
        await (self.adjust_pools_2())

        # 统一执行本时间戳内所有接口操作
        await self.execute_interface_ops(current_time)
        self.interface_ops.clear()

    async def execute_interface_ops(self,current_time):
        """
        依次执行本时间戳内待发出的所有接口调用，并将每个操作单独写入文件
        """
        with open('true.json', 'a') as logfile:
            for op in self.interface_ops:
                log_entry = {}
                if op[0] == "remove":
                    gpu = op[1]

                    log_entry = {"timestamp":current_time,"operation": "remove", "params": [gpu]}
                elif op[0] == "add":
                    gpu = op[2]
                    actor = op[1]

                    log_entry = {"timestamp":current_time,"operation": "add", "params": [actor, gpu]}
                elif op[0] == "change_engine":
                    gpu, from_role, new_gpu, to_role = op[1], op[2], op[3], op[4]
                    log_entry  = [
        {"timestamp":current_time,"operation": "change", "params": [from_role, gpu, to_role, new_gpu]}]
                else:
                    print("Unknown op:", op)

                json.dump(log_entry, logfile)
                logfile.write(",\n")


    def print_status(self):
        #print("【GPU time】", self.gpu_lifetime)
        print("【 GPU】", self.active_gpus)
        print("【prefill 】", self.prefill_pool)
        print("【decode 】", self.decode_pool)
        print("=====================================")


# 示例主程序
async def main():
    n = 0
    a = 1.7  
    manager = GPUResourceManager(n, a)

    # 模拟事件序列：
    for event in events1:
        print(f" {event['time']} ...")
        await manager.process_event(event)
        manager.print_status()
    print(manager.prefillremove)
    print(manager.decoderemove)
if __name__ == "__main__":
    asyncio.run(main())
