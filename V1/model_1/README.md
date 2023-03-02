



输入是 static feature，从固定的预训练模型中提取出来；
输出就是 mouth 和 eye 的控制器；

模型架构是 encoder 和 decoder，但是 decoder 使用的同样是 transformer encoder；
 decoder 的输入直接就是 encoder 的输出，暂时没有加入其余的信息转换成的 embedding；

 











