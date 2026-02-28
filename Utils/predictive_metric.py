"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

-----------------------------


predictive_metrics.py

Note: Use Post-hoc RNN to predict one-step ahead (last feature)
"""

# Necessary Packages
import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf1
tf.compat.v1.disable_eager_execution()
import numpy as np
from sklearn.metrics import mean_absolute_error
from Utils.metric_utils import extract_time

 
def predictive_score_metrics(ori_data, generated_data):
  """Report the performance of Post-hoc RNN one-step ahead prediction.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    
  Returns:
    - predictive_score: MAE of the predictions on the original data
  """
  # Initialization on the Graph
  tf1.reset_default_graph()

  # Basic Parameters
  no, seq_len, dim = ori_data.shape
  
  # Set maximum sequence length and each sequence length
  ori_time, ori_max_seq_len = extract_time(ori_data)
  generated_time, generated_max_seq_len = extract_time(ori_data)
  max_seq_len = max([ori_max_seq_len, generated_max_seq_len]) 
  # max_seq_len = 36 
     
  ## Builde a post-hoc RNN predictive network 
  # Network parameters
  hidden_dim = int(dim/2)
  iterations = 5000
  batch_size = 128

  # Input place holders
  X = tf1.placeholder(tf.float32, [None, max_seq_len-1, dim-1], name = "myinput_x")
  T = tf1.placeholder(tf.int32, [None], name = "myinput_t")
  Y = tf1.placeholder(tf.float32, [None, max_seq_len-1, 1], name = "myinput_y")
    
  # Predictor function
  def predictor (x, t):
    """Simple predictor function.
    
    Args:
      - x: time-series data
      - t: time information
      
    Returns:
      - y_hat: prediction
      - p_vars: predictor variables
    """
    with tf1.variable_scope("predictor", reuse = tf1.AUTO_REUSE) as vs:
      p_cell = tf1.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'p_cell')
      p_outputs, p_last_states = tf1.nn.dynamic_rnn(p_cell, x, dtype=tf.float32, sequence_length = t)
      # y_hat_logit = tf.contrib.layers.fully_connected(p_outputs, 1, activation_fn=None)
      y_hat_logit = tf1.layers.dense(p_outputs, 1, activation=None)
      y_hat = tf.nn.sigmoid(y_hat_logit)
      p_vars = [v for v in tf1.all_variables() if v.name.startswith(vs.name)]
    
    return y_hat, p_vars
    
  y_pred, p_vars = predictor(X, T)
  # Loss for the predictor
  p_loss = tf1.losses.absolute_difference(Y, y_pred)
  # optimizer
  p_solver = tf1.train.AdamOptimizer().minimize(p_loss, var_list = p_vars)
        
  ## Training    
  # Session start
  sess = tf1.Session()
  sess.run(tf1.global_variables_initializer())

  from tqdm.auto import tqdm
    
  # Training using Synthetic dataset
  for itt in tqdm(range(iterations), desc='training', total=iterations):
          
    # Set mini-batch
    idx = np.random.permutation(len(generated_data))
    train_idx = idx[:batch_size]     
            
    X_mb = list(generated_data[i][:-1,:(dim-1)] for i in train_idx)
    T_mb = list(generated_time[i]-1 for i in train_idx)
    Y_mb = list(np.reshape(generated_data[i][1:,(dim-1)],[len(generated_data[i][1:,(dim-1)]),1]) for i in train_idx)        
          
    # Train predictor
    _, step_p_loss = sess.run([p_solver, p_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})
    
  ## Test the trained model on the original data
  idx = np.random.permutation(len(ori_data))
  train_idx = idx[:no]

  # idx = np.random.permutation(len(generated_data))
  # train_idx = idx[:batch_size]       
  # X_mb = list(generated_data[i][:-1,:(dim-1)] for i in train_idx)
  # T_mb = list(generated_time[i]-1 for i in train_idx)
  # Y_mb = list(np.reshape(generated_data[i][1:,(dim-1)],[len(generated_data[i][1:,(dim-1)]),1]) for i in train_idx) 
  
  X_mb = list(ori_data[i][:-1,:(dim-1)] for i in train_idx)
  T_mb = list(ori_time[i]-1 for i in train_idx)
  Y_mb = list(np.reshape(ori_data[i][1:,(dim-1)], [len(ori_data[i][1:,(dim-1)]),1]) for i in train_idx)
    
  # Prediction
  pred_Y_curr = sess.run(y_pred, feed_dict={X: X_mb, T: T_mb})
    
  # Compute the performance in terms of MAE
  MAE_temp = 0
  for i in range(no):
    MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i,:,:])
    
  predictive_score = MAE_temp / no
    
  return predictive_score

  # args =  Args_Example()
  # 从命令行获取参数
  args_parsed = parse_arguments()  # Namespace(config_path='./Config/etth1.yaml', gpu=0, save_dir='./forecasting_exp')
  # 包装到 Args_Example 类（方便后续使用）
  args = Args_Example(args_parsed.config_path, args_parsed.save_dir, args_parsed.gpu)
  # print(args),地址
  seq_len = 96  # One-shot 预测长度
  configs = load_yaml_config(args.config_path)  # 读取 YAML 配置（包含模型结构、超参数、数据设置等）
  # print(configs)

  device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')  # 选择 GPU 或 CPU

  model = instantiate_from_config(configs['model']).to(device)
  # model.use_ff = False
  # 开启 diffusion 快速采样模式（若模型支持）
  model.fast_sampling = True
  # configs['solver']['max_epochs']=100
  ###################################################
  # 构建训练集 dataloader
  ###################################################
  dataloader_info = build_dataloader(configs, args)  # print(dataloader_info) 地址
  dataloader = dataloader_info['dataloader']
  ###################################################
  # Trend conv pipeline (training + optional plotting)

  trainer = Trainer(config=configs, args=args, model=model,
                    dataloader={'dataloader': dataloader})  # 初始化 Trainer（包含优化器、损失函数、训练流程等）
  trainer.train()  # 训练模型
  ###################################################
  # -----------   推理（预测）阶段   ----------------
  ###################################################
  args.mode = 'predict'  # 预测，并非填补缺失值
  args.pred_len = seq_len  # 预测步长设置，用于 dataloader_cond
  test_dataloader_info = build_dataloader_cond(configs, args)  # 构建测试集 dataloader（有条件的数据）condition

  # 取得标准化后的测试数据
  test_scaled = test_dataloader_info['dataset'].samples
  scaler = test_dataloader_info['dataset'].scaler
  seq_length, feat_num = seq_len * 2, test_scaled.shape[-1]  # 序列长度 & 变量数
  pred_length = seq_len
  real = test_scaled
  test_dataset = test_dataloader_info['dataset']  # 提取测试 data/dataloader
  test_dataloader = test_dataloader_info['dataloader']

  sample, real_ = trainer.sample_forecast(test_dataloader, shape=[seq_len, feat_num])
  mask = test_dataset.masking  # 掩码，用于忽略某些值（若数据有缺失）

  ###################################################
  # 计算 MSE/MAE 误差
  ###################################################
  mse = mean_squared_error(sample.reshape(-1), real_.reshape(-1))
  mae = mean_absolute_error(sample.reshape(-1), real_.reshape(-1))

  print(mse, mae)
