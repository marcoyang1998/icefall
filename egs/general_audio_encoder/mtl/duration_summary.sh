#!/bin/bash

# 输入文件夹
logdir=$1

# 临时文件存储每个batch的聚合数据
tmpfile=$(mktemp)

num_logs=$(ls $logdir | wc -l)
echo "num_logs=[$num_logs]"
# exit

# 提取所有log文件中的task cuts行，处理并输出到临时文件
grep "task cuts" "$logdir"/* | gawk -v nlog=8 '
{
    match($0, /batch ([0-9]+): task cuts: ([0-9]+), ([0-9]+), task durations: ([0-9.]+), ([0-9.]+)/, m);
    if (m[1] != "") {
        batch = m[1];
        cut1 = m[2]; cut2 = m[3];
        dur1 = m[4]; dur2 = m[5];

        sum_cut1[batch] += cut1;
        sum_cut2[batch] += cut2;
        sum_dur1[batch] += dur1;
        sum_dur2[batch] += dur2;
        count[batch]++;

        dur1_list[batch] = dur1_list[batch] " " dur1;
        dur2_list[batch] = dur2_list[batch] " " dur2;
    }
}
END {
    for (b in sum_cut1) {
        printf "%d %.0f %.0f %.3f %.3f\n", b, sum_cut1[b], sum_cut2[b], sum_dur1[b], sum_dur2[b] >> "'"$tmpfile"'"
    }
}' 

# 对输出按 batch id 排序并打印
sort -n "$tmpfile"

# 计算总的 duration 平均和方差
awk '
{
    dur1_sum += $4;
    dur2_sum += $5;
    dur1_sq_sum += $4 * $4;
    dur2_sq_sum += $5 * $5;
    n++;
}
END {
    mean1 = (dur1_sum / n) ;
    mean2 = (dur2_sum / n) ;
    var1 = (dur1_sq_sum / n) - (mean1 * mean1);
    var2 = (dur2_sq_sum / n) - (mean2 * mean2);
    printf "\nAverage duration1: %.3f, duration2: %.3f\n", mean1, mean2;
    printf "Variance duration1: %.3f, duration2: %.3f\n", var1, var2;
}' "$tmpfile"

# 删除临时文件
rm "$tmpfile"