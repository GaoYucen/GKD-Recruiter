"""
The double DQN based on this paper: https://arxiv.org/abs/1509.06461

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""
import psutil
import time
import os
import ENV1_2 as EN
import numpy as np
from copy import deepcopy
from random import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1)
tf.set_random_seed(1)

# Data = r"/content/drive/MyDrive/data/transNetHEPT_1.txt"
# # testData = r"/content/drive/MyDrive/data/DBLP.txt"
# model_address = r'/content/drive/MyDrive/models_test/'
# log = open(r"/content/drive/MyDrive/V&A/ExpRes/VA_BDQN_hep.txt","w")

filename = list()
filepath = "E:\Research\paper\\2022crowdsensing\data\input_5000_0\\"
logpath = "E:\Research\paper\\2022crowdsensing\log\\input_5000_0\\random\\"

budget_tot = 100 # k
task_num = 100
budget = 1

# data_list = ["\\task0", "\\task1", "\\task2"]
task_list = list()
task_dict = np.zeros([2, task_num])

for i in range(task_num):
    task_name = "task" + str(i)
    task_list.append(task_name)

data_list = task_list
model_name = "_ICSOC"
inf_task = 0
time_task = 0

## hyperparameters
# budget = 10 # 50 seed set capacity
Greedy = 0.2 # P to apply policy (if no we take random action)
Memory_size = budget*2 # replay buffer capacity(total)
learning_frequency = 100
penalty = 0 # reward for dropping a node
Round = 1 # 20
traing_turns = 20
process = psutil.Process(os.getpid())

class DoubleDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=Greedy,
            replace_target_iter=20,
            memory_size=Memory_size,
            batch_size=40,
            e_greedy_increment=None,
            output_graph=False,
            double_q=True,
            sess=None,
            lam=0.5, # lambda_0
            delta_lam=0.1,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # BDQN
        self.total_record_accept = list()
        self.total_record_reject = list()
        self.memory_counter_acc = 0
        self.memory_counter_rej = 0
        self.lam = lam
        self.delta_lam = delta_lam
        self.lam_min = 0

        self.double_q = double_q    # decide to use double q or not

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+2))

        self._build_net()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

        # print("Loading ...")
        # self.restore_model(model_address)
        # print("Loading Complete!")

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l1, w2) + b2
            return out
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def restore_model(self,address):
        '''
        Restore the model parameters trained before
        Input: address -- address of model
        '''
        tf.train.Saver().restore(self.sess, address)
        print("model restored")

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        if a == 1:  # accepted
            index = int(self.memory_counter_acc % (self.memory_size / 2))
            self.memory[index, :] = transition
            self.memory_counter_acc += 1
            print("Transition completed. Node accepted")
        else:  # reject
            index = int(self.memory_counter_rej % (self.memory_size / 2) + (self.memory_size / 2))
            self.memory[index, :] = transition
            self.memory_counter_rej += 1
            print("Transition completed. Node rejected")
        self.memory_counter +=1

    def choose_action_ran(self, feat_cur):
        observation = np.mat(feat_cur)
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)

        if not hasattr(self, 'q'):  # record action value it gets
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:  # choosing action
            action = np.random.randint(0, self.n_actions)
        print("action:", action)
        return action

    def choose_action(self, feat_cur):
        observation = np.mat(feat_cur)
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)

        if not hasattr(self, 'q'):  # record action value it gets
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        print("action:", action)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter_acc + self.memory_counter_rej > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],    # next observation
                       self.s: batch_memory[:, -self.n_features:]})    # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next = np.max(q_next, axis=1)    # the natural DQN

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

RL = DoubleDQN(2, 3, output_graph=False)

def run_it(Data):
    TRAIN = 1
    step = 0
    action_times = 0  # times we have choose actions
    total_reward = list()
    G = EN.Env(Data, budget)
    print("G completed")

    episode = 0
    inf_tot = 0
    t = 0.0
    start_M = process.memory_info().rss
    # check_M = 0

    # h = int(budget / 10) +1
    # for i in range(h):
    #    G.greedy(i*10)

    next = np.zeros(2) # store s_

    while(episode < Round):
        print("episode: ", episode)
        if episode >= traing_turns:
            print("# -------------------WARNING: From training to Testing !---------------", flush=True)
            TRAIN = 0
            test = deepcopy(G)
            print("Graph copied. TRAIN = 0")
        else:
            TRAIN = 1
            test = deepcopy(G)
        Inf = []
        Col = []
        inf_max = test.list[0][1]
        step_c = 0
        print("starting selection")

        while (True):

            if len(test.seed) == budget:
                print(len(test.seed), ' ', EN.IC(test.graph, test.seed), ' ', t, ' ', file=log, flush=True)
                print(test.seed, file=log, flush=True)
                print("memory used", process.memory_info().rss - start_M, file=log, flush=True)

            start = time.time()
            input_at_step_test = []

            # Budget Fulfil
            print("current seed num:", len(test.seed))
            inf_overall, inf_col = test.node2feat(step_c)
            inf_overall = test.list[step_c][1]
            if len(test.seed) == budget:
                inf_total = EN.IC(test.graph, test.seed)
                print("Current influence of the seed set is:", inf_total)
                break
            elif inf_overall < RL.lam * inf_max or step_c + 100 > len(test.list):
                print("Change agent")

                if RL.lam > RL.lam_min:
                    RL.lam = RL.lam - RL.delta_lam
                    print("Lambda decreased, now trying to start over")
                    step_c = 0
                    inf_overall, inf_col = test.node2feat(step_c)
                    inf_overall = test.list[step_c][1]
                    # 这里考虑一下step c直接到0还是退出循环 按新的agent直接从头选
                else:
                    print("Error! Unable to select a full seed set")
                    break

            if TRAIN == 1:
                action = RL.choose_action_ran(test.netInput)
            else:
                action = RL.choose_action(test.netInput)
            print("action: ",action, "Occupied budget: ", len(test.seed))

            if action == 1: #accepted\
                print("Round: ",episode ,"Current node: ",step_c, " accepted ")
                input_at_step_test = test.netInput
                reward = (test.steps(step_c,1,TRAIN) - inf_col * RL.lam)
                test.list.pop(step_c)
                #reward = (test.steps(step_c,1,TRAIN) - L[step_c + 1][1] * (sigmoid - test.seed_size()/budget*0.5))
                #reward = (test.steps(step_c,1,TRAIN) - L[step_c + 1][1] * sigmoid)
                step_next = step_c + 1
            else:
                print("Round: ",episode ,"Current node: ",step_c, " rejected")
                input_at_step_test = test.netInput
                reward = penalty
                #reward = (L[step_c + 1][1] * sigmoid - test.steps(step_c,0,TRAIN))
                #reward =
                step_next = step_c + 1

            test.node2feat(step_next)
            # next = test.netInput
            print("The reward is: ", reward, flush=True)
            # if episode > 0:
            RL.store_transition(input_at_step_test, action, reward, test.netInput)

            action_times += 1

            if TRAIN == 1:
                if(action_times > budget/2) and (action_times % learning_frequency ==0):
                    print(".................................Learning ...............................", flush=True)
                    RL.learn()

            step_c = step_next
            print("current selection complete")
            t += time.time() - start

        # saver = tf.train.Saver()
        # print("Saving ...................")
        # if TRAIN == 1 and episode%5 == 0:
        #     saver.save(RL.sess, r"/content/drive/MyDrive/models/")
        #     print("model saved")
        inf = EN.IC(test.graph, test.seed)
        print("Current influence is: ", inf)
        inf_tot += inf
        episode += 1
    end_M = process.memory_info().rss  # in bytes
    print("average influence spread:", inf_tot / Round)
    print("average time:", t / Round)
    print("memory cost:", (end_M - start_M) / Round)
    # print("average influence spread:", inf_tot / Round, file=log, flush=True)
    # print("average time:", t / Round, file=log, flush=True)
    # print("memory cost:", (end_M - start_M) / Round, file=log, flush=True)
    print('Complete!')
    return inf_tot / Round, t / Round

# count = 0 # only for testing

log_path = logpath + str(budget_tot) + model_name + "_1.txt"
# log_path = log_path + model_name + ".txt"
print(log_path)

log = open(log_path, "w")
for _ in range(budget_tot):
    k = randint(0, task_num-1)
    data = data_list[k]
    # count += 1
    file = filepath + data + ".txt"

    # log_path = logpath + data + ".txt"
    # OtherTime(file, test.graph, budget)
    inf_, t_ = run_it(file)
    inf_task += inf_
    time_task += t_
    task_dict[0][k] += 1
    task_dict[1][k] += inf_

# run_it()
log_tot = logpath + str(budget_tot) + model_name + "_tot_1.txt"
log1 = open(log_tot, "w")
print(budget_tot, ' ', inf_task, ' ', time_task, ' ', file=log1, flush=True)
print(*task_dict[0], file=log1, flush=True)
print(*task_dict[1], file=log1, flush=True)
log1.close()