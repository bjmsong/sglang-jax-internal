import timeit
import functools
import numpy as np
import jax
import jax.numpy as jnp
from matmul import matmul

v6e_flops = 926e12

def matmul_flops(m: int, k: int, n: int):
    return 2 * m * k * n

def benchmark(f, ntrials: int = 50):
    def run(*args, **kwargs):
        # Compile function first
        jax.block_until_ready(f(*args, **kwargs))
        # Time function
        result = timeit.timeit(lambda: jax.block_until_ready(f(*args, **kwargs)),
                               number=ntrials)
        time = result / ntrials
        return time
    return run

def analyze_matmul(m: int, k: int, n: int, dtype: np.dtype, mm_func):
    x = jnp.ones((m, k), dtype=dtype)
    y = jnp.ones((k, n), dtype=dtype)
    time = benchmark(mm_func)(x, y)
    mm_flops = matmul_flops(m, k, n) / time
    utilization = mm_flops / v6e_flops * 100
    return time, mm_flops, utilization

def test_configuration(bm, bk, bn, matrix_sizes):
    """测试特定的bm, bk, bn配置"""
    try:
        mm = functools.partial(matmul, bm=bm, bk=bk, bn=bn)
        results = {}
        
        for m, k, n in matrix_sizes:
            time, flops, utilization = analyze_matmul(m, k, n, jnp.bfloat16, mm)
            results[f"{m}x{k}x{n}"] = {
                'time': time,
                'flops': flops,
                'utilization': utilization
            }
        
        # 计算平均利用率
        avg_utilization = np.mean([r['utilization'] for r in results.values()])
        return results, avg_utilization
    except Exception as e:
        print(f"配置 bm={bm}, bk={bk}, bn={bn} 失败: {e}")
        return None, 0

def optimize_tile_sizes():
    """遍历不同的tile size组合找到最优配置"""
    
    # 定义要测试的矩阵大小
    matrix_sizes = [
        (1024, 1024, 1024),
        (4096, 4096, 4096),
        (8192, 8192, 8192)
    ]
    
    # 定义要测试的tile size范围
    tile_sizes = [512, 1024, 2048, 4096]
    
    best_config = None
    best_utilization = 0
    all_results = []
    
    print("开始遍历bm, bk, bn组合...")
    print("=" * 80)
    
    total_combinations = len(tile_sizes) ** 3
    current_combination = 0
    
    for bm in tile_sizes:
        for bk in tile_sizes:
            for bn in tile_sizes:
                current_combination += 1
                print(f"进度: {current_combination}/{total_combinations} - 测试 bm={bm}, bk={bk}, bn={bn}")
                
                results, avg_utilization = test_configuration(bm, bk, bn, matrix_sizes)
                
                if results is not None:
                    all_results.append({
                        'bm': bm,
                        'bk': bk,
                        'bn': bn,
                        'results': results,
                        'avg_utilization': avg_utilization
                    })
                    
                    if avg_utilization > best_utilization:
                        best_utilization = avg_utilization
                        best_config = (bm, bk, bn)
                        print(f"发现更好的配置: bm={bm}, bk={bk}, bn={bn}, 平均利用率: {avg_utilization:.2f}%")
    
    # 按平均利用率排序
    all_results.sort(key=lambda x: x['avg_utilization'], reverse=True)
    
    print("\n" + "=" * 80)
    print("最优配置结果:")
    print("=" * 80)
    
    # 显示前10个最佳配置
    for i, result in enumerate(all_results[:10]):
        bm, bk, bn = result['bm'], result['bk'], result['bn']
        avg_util = result['avg_utilization']
        print(f"排名 {i+1}: bm={bm}, bk={bk}, bn={bn}, 平均利用率: {avg_util:.2f}%")
        
        # if i == 0:  # 显示最优配置的详细结果
        print("\n详细结果:")
        for size_name, metrics in result['results'].items():
            print(f"  {size_name}: 时间={metrics['time']:.6f}s, "
                    f"TFLOPS={metrics['flops']/1e12:.2f}, "
                    f"利用率={metrics['utilization']:.2f}%")
    
    return best_config, all_results

if __name__ == "__main__":
    best_config, all_results = optimize_tile_sizes()
    
    print(f"\n最优配置: bm={best_config[0]}, bk={best_config[1]}, bn={best_config[2]}")
    
    # 保存结果到文件（如果文件已存在则覆盖）
    with open('optimization_results.txt', 'w') as f:
        f.write("Tile Size Optimization Results\n")
        f.write("=" * 50 + "\n\n")
        
        for i, result in enumerate(all_results[:20]):  # 保存前20个结果
            bm, bk, bn = result['bm'], result['bk'], result['bn']
            avg_util = result['avg_utilization']
            f.write(f"排名 {i+1}: bm={bm}, bk={bk}, bn={bn}, 平均利用率: {avg_util:.2f}%\n")
            
            if i == 0:
                f.write("\n最优配置详细结果:\n")
                for size_name, metrics in result['results'].items():
                    f.write(f"  {size_name}: 时间={metrics['time']:.6f}s, "
                           f"TFLOPS={metrics['flops']/1e12:.2f}, "
                           f"利用率={metrics['utilization']:.2f}%\n")
    
    print(f"\n结果已保存到 optimization_results.txt")
