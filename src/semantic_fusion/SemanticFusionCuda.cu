/*
 * This file is part of SemanticFusion.
 *
 * Copyright (C) 2017 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is SemanticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/semantic-fusion/semantic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#include <stdio.h>
#include <assert.h> 

#include <cuda_runtime.h>

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool
        abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    } 
}

/*
@ brief
更新概率图和最大概率图
就是论文里的贝叶斯更新

@ param
ids：map中surfel的ids。map->GetSurfelIdsGpu()
ids_width：map->width
ids_height = map->height();
probabilities：分割结果blob，只读
prob_width = probs->width();
prob_height = probs->height();
prob_channels = probs->channels();
map_table：class_pro，所有surfel所有类别的概率，待更新。class_probabilities_gpu_->mutable_gpu_data()，可写
map_max：class_max，最大概率的map，可写。class_max_gpu_->mutable_gpu_data()
map_size：现在class_pro中的surfel数量。class_probabilities_gpu_->width()
*/
__global__ 
void semanticTableUpdate(cudaTextureObject_t ids, const int ids_width, const int ids_height, 
                          const float* probabilities, const int prob_width, const int prob_height, 
                          const int prob_channels,float* map_table,float* map_max,
                          const int map_size)
{
    // 当前处理的index？
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    // New uniqueness code
    // 在16*16的区域内搜索，定义搜索范围
    const int check_patch = 16;
    const int x_min = (x - check_patch) < 0 ? 0 : (x - check_patch);
    const int x_max = (x + check_patch) > 640 ? 640 : (x + check_patch);
    const int y_min = (y - check_patch) < 0 ? 0 : (y - check_patch);
    int surfel_id = tex2D<int>(ids,x,y);
    int first_h, first_w;
    // 在范围内找到第一个id相等的坐标
    for (int h = y_min; h < 480; ++h) {
        for (int w = x_min; w < x_max; ++w) {
            int other_surfel_id = tex2D<int>(ids,w,h);
            if (other_surfel_id == surfel_id) {
                first_h = h;
                first_w = w;
                break;
            }
        }
    }
    // 检查在不在map里面吗
    if (first_h != y || first_w != x) {
        surfel_id = 0;
    }
    if (surfel_id > 0) {
        // x，y是在map里面的坐标，转到图像上的坐标。即找到surfel对应的像素点
        const int prob_x = static_cast<int>((float(x) / ids_width) * prob_width);
        const int prob_y = static_cast<int>((float(y) / ids_height) * prob_height);
        // 到下一个channel要加的offset，也是图像中的像素个数
        const int channel_offset = prob_width * prob_height;
        // 当前第k帧这个像素的概率（第一个类别）
        const float* probability = probabilities + (prob_y * prob_width + prob_x);
        // 前k-1帧对应surfel的概率（第一个类别）
        float* prior_probability = map_table + surfel_id;
        float total = 0.0;
        // 循环所有类别，累积所有类别前k-1帧概率乘第k帧概率的结果，累加得到total，
        // 这时候prior_probability和probability都指向了最后一个类别的位置
        // 这一步是为了算出来公式里的归一化因子z，并算出各类别乘积的结果
        for (int class_id = 0; class_id < prob_channels; ++class_id) {
            // 
            prior_probability[0] *= probability[0];
            total += prior_probability[0];
            probability += channel_offset;
            prior_probability += map_size;
        }
        // Reset the pointers to the beginning again
        // 把他们重置到最开始的位置（第一个类别）
        probability = probabilities + (prob_y * prob_width + prob_x);
        prior_probability = map_table + surfel_id;
        float max_probability = 0.0;
        int max_class = -1;
        float new_total = 0.0; // 没用到
        // 再次循环所有类别，更新概率图
        // 这一步是为了归一化并得到最大值
        for (int class_id = 0; class_id < prob_channels; ++class_id) {
            // Something has gone unexpectedly wrong - reinitialse
            if (total <= 1e-5) {
                // 若总概率值太小，就所有类别均分概率
                prior_probability[0] = 1.0f / prob_channels;
            } else {
                // 除以归一化因子
                prior_probability[0] /= total;
                // 更新最大的概率值和类别
                if (class_id > 0 && prior_probability[0] > max_probability) {
                    max_probability = prior_probability[0];
                    max_class = class_id;
                }
            }
            // 指向下一个类别
            new_total += prior_probability[0];
            probability += channel_offset;
            prior_probability += map_size;
        }
        // 更新最大的概率。 class_map
        map_max[surfel_id] = static_cast<float>(max_class);
        map_max[surfel_id + map_size] = max_probability;
        map_max[surfel_id + map_size + map_size] += 1.0;
    }
}

/*
@param
ids：map中surfel的ids。map->GetSurfelIdsGpu()
ids_width：map->width 显示图像的长宽？
ids_height = map->height();
probabilities：分割结果blob，只读
prob_width = probs->width(); 这个width是整张图的width <w,h,c,n> n是1 ？
prob_height = probs->height(); 网络输出层
prob_channels = probs->channels();
map_table：class_pro，所有surfel所有类别的概率，待更新。class_probabilities_gpu_->mutable_gpu_data()，可写
map_max：class_max，最大概率的map，可写。class_max_gpu_->mutable_gpu_data()
map_size：现在class_pro中的surfel数量。class_probabilities_gpu_->width()
*/
__host__ 
void fuseSemanticProbabilities(cudaTextureObject_t ids, const int ids_width, const int ids_height, 
                          const float* probabilities, const int prob_width, const int prob_height, 
                          const int prob_channels,float* map_table, float* map_max,
                          const int map_size)
{
    // NOTE Res must be pow 2 and > 32
    const int blocks = 32;
    dim3 dimGrid(blocks,blocks); // 32*32 的 grid
    dim3 dimBlock(640/blocks,480/blocks); // 每一个block中分得 (640/32, 480/32)
    semanticTableUpdate<<<dimGrid,dimBlock>>>(ids,ids_width,ids_height,probabilities,
        prob_width,prob_height,prob_channels,map_table,map_max,map_size);
    gpuErrChk(cudaGetLastError());
    gpuErrChk(cudaDeviceSynchronize());
}

__global__ 
/*
@ biref
看起来这个函数并没有概率累乘？ 只是在更新table，加入新的surfel，删除旧的surfel(不知道有没有删除)

@ param
n：要更新的数量，surfel的数量乘以pro的height。num_to_update = new_prob_width * prob_height
deleted_ids：要删除的surfel id，指针
num_deleted：要删除的surfel数量
current_table_size：当前的surfel数
probability_table：class_pro的blob数据，只读
prob_width：应该是surfel的数量，n？class_probabilities_gpu_->width()
prob_height：类别的数量，c。class_probabilities_gpu_->height()  prob height is the number of classes
new_prob_width：新map里所有surfle的数量。map->GetMapSurfelCount()
new_probability_table：可写的class_pro_buffer。用于存储新的
map_table：class_max,只读
new_map_table：class_max_buffer，可写
*/
void updateTable(int n, const int* deleted_ids, const int num_deleted, const int current_table_size,
                 float const* probability_table, const int prob_width, const int prob_height, 
                 const int new_prob_width, float* new_probability_table, float const * map_table, float* new_map_table)
{
    // 更新的surfel的指针，因为GPU并行？
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        // index： row： classes col：components
        // 属于哪个类别 行数
        // c*n的矩阵
        // 从0开始？
        const int class_id = index / new_prob_width;
        // 属于这个类别的哪个component  列数
        // 从1开始？
        const int component_id = index - (class_id * new_prob_width);
        // 为什么newid用的是prob_width，而前面的id用的是new prob width?
        // 这里是不是old_id更恰当？ 得到的是这个类别这个surfel在原来map里的id
        const int new_id = (class_id * prob_width) + component_id;
        // 不在删除的范围内，即，是一个新的surfel？ 
        // 新的概率值写入buffer
        if (component_id >= num_deleted) {
            // Initialise to prior (prob height is the number of classes)
            // 建一个新节点，各类概率等分
            new_probability_table[new_id] = 1.0f / prob_height;
            // Reset the max class surfel colouring lookup
            // 这个id位置对应的三行，为负表示不存在最大的
            new_map_table[component_id] = -1.0;
            new_map_table[component_id + prob_width] = -1.0;
            new_map_table[component_id + prob_width + prob_width] = 0.0;
        }
        // 若已经存在，则更新 
        // 同一个surfel，在图里的index不一样，但surfel总数不变？？
        else {
            // 原class_pro中的component编号
            int offset = deleted_ids[component_id];
            // 赋值为原来它的概率
            new_probability_table[new_id] = probability_table[(class_id * prob_width) + offset];
            // Also must update our max class mapping
            // 更新class_max
            // 直接赋值
            new_map_table[component_id] = map_table[offset];
            new_map_table[component_id + prob_width] = map_table[prob_width + offset];
            new_map_table[component_id + prob_width + prob_width] = map_table[prob_width + prob_width + offset];
        }
    }
}

/*
@ param
filtered_ids：*int，map里要删除的surfel id，由map->GetDeletedSurfelIdsGpu()获得 // 为什么删除？
num_filtered：map里要删除surfel数量，由map->GetMapSurfelDeletedCount()获得
current_table_size：当前table大小，当前surfel的数量，每次调用这个函数后更新
probability_table：class_pro的gpu数据（BLOB），所有点所有类别的概率图。class_probabilities_gpu_->gpu_data()获得。是一个只读的
prob_width：图里点的数量。class_probabilities_gpu_->width()，(w,h,c,n)的blob，w是1？？
prob_height：类别数。class_probabilities_gpu_->height()，h也是1？？
new_prob_width：新的map里所有surfle的数量。map->GetMapSurfelCount()。
new_probability_table：新的table，写入buffer。class_probabilities_gpu_buffer_->mutable_gpu_data()，mutable意味着可写
map_table：保存每个surfel当前概率最大的类别的table。class_max_gpu_->gpu_data()，只读。
new_map_table：新的table，class_max_gpu_buffer_->mutable_gpu_data()，可写。
*/
__host__ 
void updateProbabilityTable(int* filtered_ids, const int num_filtered, const int current_table_size,
                            float const* probability_table, const int prob_width, const int prob_height, 
                            const int new_prob_width, float* new_probability_table, 
                            float const* map_table, float* new_map_table)
{
    const int threads = 512;
    // 待更新的数量=新的map中点的数量*类别数。即所有点的所有概率都要更新
    const int num_to_update = new_prob_width * prob_height;
    // 每个线程要负责多少个
    const int blocks = (num_to_update + threads - 1) / threads;
    // dim3:三维向量(x,y,z)
    // GPU的某种设定
    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    updateTable<<<dimGrid,dimBlock>>>(num_to_update,filtered_ids,num_filtered,current_table_size,
        probability_table,prob_width,prob_height,new_prob_width,new_probability_table, 
        map_table, new_map_table);
    gpuErrChk(cudaGetLastError());
    gpuErrChk(cudaDeviceSynchronize());
}

/*
@ brief
把class_pro中的概率投影到render_map上，render map是和elastic fusion的map一样的，用于最后的可视化
@ param
ids：map里的surfelid 。map->GetSurfelIdsGpu()
ids_width：id_width = map->width()
ids_height：id_height = map->height()
probability_table：class_pro，可写。class_probabilities_gpu_->mutable_gpu_data()
prob_width：n
table_height：c
rendered_probabilities：rendered_class_probabilities_gpu_->mutable_gpu_data()
*/
__global__ 
void renderProbabilityMapKernel(cudaTextureObject_t ids, const int ids_width, const int ids_height, 
                          const float* probability_table, const int prob_width, const int prob_height, 
                          float* rendered_probabilities) 
{
    // 当前处理的surfel
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int surfel_id = tex2D<int>(ids,x,y);
    // surfel的索引
    int projected_probability_offset = y * ids_width + x;
    int probability_table_offset = surfel_id;
    // 对于所有类别
    for (int class_id = 0; class_id < prob_height; ++class_id) {
        // 如果这个点存在
        if (surfel_id > 0) {
            // 把class_pro中该surfel对应类别的概率赋给render_pro
            rendered_probabilities[projected_probability_offset] = probability_table[probability_table_offset];
        } 
        else 
        // 否则，render_pro中该位置的概率是1或0。仅第0个类别为1，其余类别为0。
        {
            rendered_probabilities[projected_probability_offset] = ((class_id == 0) ? 1.0 : 0.0);
        }
        // 指向下一个类别的位置。
        projected_probability_offset += (ids_width * ids_height);
        probability_table_offset += prob_width;
    }
}

/*
class_pro概率图投影到render map上。通过这个接口转给GPU运行。
*/
__host__
void renderProbabilityMap(cudaTextureObject_t ids, const int ids_width, const int ids_height, 
                          const float* probability_table, const int prob_width, const int prob_height, 
                          float* rendered_probabilities) 
{
    // NOTE Res must be pow 2 and > 32
    const int blocks = 32;
    dim3 dimGrid(blocks,blocks);
    dim3 dimBlock(ids_width/blocks,ids_height/blocks);
    renderProbabilityMapKernel<<<dimGrid,dimBlock>>>(ids,ids_width,ids_height,probability_table,prob_width,prob_height,rendered_probabilities);
    gpuErrChk(cudaGetLastError());
    gpuErrChk(cudaDeviceSynchronize());
}

/*
@ brief
更新最大概率值和类别
*/
__global__ 
void updateMaxClassKernel(const int n, const float* probabilities, const int classes,
                          float* map_max, const int map_size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        // Reset the pointers to the beginning again
        const float* probability = probabilities + index;
        probability += map_size;
        float max_probability = 0.0;
        int max_class = -1;
        for (int class_id = 1; class_id < classes; ++class_id) {
            if (probability[0] > max_probability) {
                max_probability = probability[0];
                max_class = class_id;
            }
            probability += map_size;
        }
        map_max[index] = static_cast<float>(max_class);
        map_max[index + map_size] = max_probability;
    }
}

/*
CRF里调用的，更新最大值
*/
__host__ 
void updateMaxClass(const int n, const float* probabilities, const int classes,
                    float* map_max, const int map_size)
{
    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;
    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    updateMaxClassKernel<<<dimGrid,dimBlock>>>(n,probabilities,classes,map_max,map_size);
    gpuErrChk(cudaGetLastError());
    gpuErrChk(cudaDeviceSynchronize());
}
