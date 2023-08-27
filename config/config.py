# -*- coding: utf-8 -*-
# @Time    : 2021/10/11 21:24
# @Author  : Naynix
# @File    : config.py
import configparser
import os
import logging

class Config(object):
    def __init__(self, dataset, version, logfile="main_experiment", server="local", early=0, interval=0, interval_num=0):
        cnf = configparser.ConfigParser()
        if server == "local":
            cnf_file = "/mntc/yxy/MDPP/config/cnf.ini"
        elif server == "school":
            cnf_file = "/seu_share/home/seu9yxy/yxy/main_v0/config/cnf.ini"
        else:
            cnf_file = "/nfs/users/lijinze/whd/MDPP_versions/cnf.ini"
        try:
            cnf.read(cnf_file)
        except:
            print("can't read %s successfully" % cnf_file)

        # 数据文件参数设置
        InputData = "InputData_%s" % server
        self.datafolder = cnf.get(InputData, "datafolder")
        self.dataset = dataset
        self.datafolder = os.path.join(self.datafolder, dataset)
        self.originfolder = os.path.join(self.datafolder, cnf.get(InputData, "originfolder"))
        self.vocabfile = os.path.join(self.datafolder, cnf.get(InputData, "vocabfile_" + self.dataset))
        self.processed_data = os.path.join(self.datafolder, cnf.get(InputData, "processed_data") + "_" + str(version))
        if self.dataset in ["twitter15", "twitter16"]:
            self.source_file = os.path.join(self.datafolder, cnf.get(InputData, "source_file"))
            self.tree_file = os.path.join(self.datafolder, cnf.get(InputData, "tree_file"))
            self.label_file = os.path.join(self.datafolder, cnf.get(InputData, "label_file"))

        # PP模型用
        self.userfolder = os.path.join(self.datafolder, cnf.get(InputData, "userfolder"))
        if not os.path.exists(self.userfolder):
            os.makedirs(self.userfolder)
        self.usermapfile = os.path.join(self.userfolder, cnf.get(InputData, "usermap"))
        self.followfile = os.path.join(self.userfolder, cnf.get(InputData, "follow"))
        self.retweetfile = os.path.join(self.userfolder, cnf.get(InputData, "retweet"))
        self.user_embedding_file = os.path.join(self.userfolder, cnf.get(InputData, "user_embedding"))

        self.truefolder = os.path.join(self.datafolder, cnf.get(InputData, "truefolder"))
        if not os.path.exists(self.truefolder):
            os.makedirs(self.truefolder)
        self.pop_file = os.path.join(self.truefolder, "pop_final.pkl")
        self.pop_user_early_file = os.path.join(self.truefolder, "pop_user_early_" + str(early) + ".pkl")
        self.pop_user_final_file = os.path.join(self.truefolder, "pop_user_final.pkl")
        if early == 0:
            self.tagsfile = os.path.join(self.datafolder, f"tags.pkl")
        else:
            self.tagsfile = os.path.join(self.datafolder, f"tags_{str(early)}.pkl")


        if early != 0:
            self.datafolder = os.path.join(self.datafolder, 'early')
        self.fakefolder = os.path.join(self.datafolder, cnf.get(InputData, "fakefolder"))
        self.nonfakefolder = os.path.join(self.datafolder, cnf.get(InputData, "nonfakefolder"))
        if early != 0:
            self.fakefolder = "%s_%s" % (self.fakefolder, str(early))
            self.nonfakefolder = "%s_%s" % (self.nonfakefolder, str(early))


        # 输出文件参数设置
        OutputData = "OutputData_%s" % server
        self.outputfolder = cnf.get(OutputData, "outputfolder")

        # 日志文件参数
        self.logfolder = os.path.join(self.outputfolder, cnf.get(OutputData, "logfolder"))
        self.logfolder = os.path.join(self.logfolder, self.dataset)
        if not os.path.exists(self.logfolder):
            os.makedirs(self.logfolder)
        logging.basicConfig(level=logging.INFO,
                            filename=os.path.join(self.logfolder, logfile),
                            filemode='a',
                            format='%(asctime)s - %(levelname)s: %(message)s'
                            )
        self.logger = logging.getLogger(__name__)

        # MD模型 结果保存文件
        if early == 0:
            self.result_file = os.path.join(self.logfolder, str(version) + ".csv")
        else:
            self.result_file = os.path.join(self.logfolder, str(version) + "_" + str(early) + ".csv")
        # self.result_file = os.path.join(self.logfolder, str(version) + ".csv")
        # PP模型 结果保存文件
        self.PP_result_file = os.path.join(self.logfolder, "PP_result.csv")

        #模型文件
        self.modelfolder = os.path.join(self.outputfolder, cnf.get(OutputData, "modelfolder"))
        self.modelfolder = os.path.join(self.modelfolder, self.dataset)
        if not os.path.exists(self.modelfolder):
            os.makedirs(self.modelfolder)

        #MD模型->节点表示文件
        self.reprfolder = os.path.join(self.outputfolder, cnf.get(OutputData, "reprfolder"))
        self.reprfolder = os.path.join(self.reprfolder, self.dataset)
        if not os.path.exists(self.reprfolder):
            os.makedirs(self.reprfolder)

        # MD模型参数设置
        MD_para = "MD_Para_%s" % self.dataset
        self.interval = cnf.getint(MD_para, "interval")
        self.interval_num = cnf.getint(MD_para, "interval_num")
        self.max_seq_len = cnf.getint(MD_para, "max_seq_len")
        self.max_post_num = cnf.getint(MD_para, "max_post_num")
        self.input_dim = cnf.getint(MD_para, "input_dim")
        self.user_dim = cnf.getint(MD_para, "user_dim")
        self.output_dim = cnf.getint(MD_para, "output_dim")
        self.hidden_dim = cnf.getint(MD_para, "hidden_dim")
        self.dropout = cnf.getfloat(MD_para, "dropout")
        self.nheads = cnf.getint(MD_para, "nheads")
        self.gat_layer_num = cnf.getint(MD_para, "gat_layer_num")
        self.lstm_layer_num = cnf.getint(MD_para, "lstm_layer_num")
        self.pool = cnf.get(MD_para, "pool")
        self.alpha = cnf.getfloat(MD_para, "alpha")
        self.lr = cnf.getfloat(MD_para, "lr")
        self.batch_size = cnf.getint(MD_para, "batch_size")
        self.epoch = cnf.getint(MD_para, "epoch")
        self.tunit = cnf.getint(MD_para, "tunit")

        if interval != 0:
            self.interval = interval
        if interval_num != 0:
            self.interval_num = interval_num

        # PP模型参数设置
        PP_para = "PP_Para_%s" % self.dataset
        self.PP_max_seq_len = cnf.getint(PP_para, "PP_max_seq_len")
        self.PP_max_post_num = cnf.getint(PP_para, "PP_max_post_num")
        self.PP_input_dim = cnf.getint(PP_para, "PP_input_dim")
        self.MD_input_dim = cnf.getint(PP_para, "MD_input_dim")
        self.PP_user_dim = cnf.getint(PP_para, "PP_user_dim")
        self.PP_output_dim = cnf.getint(PP_para, "PP_output_dim")
        self.PP_hidden_dim = cnf.getint(PP_para, "PP_hidden_dim")
        self.PP_lstm_layer_num = cnf.getint(PP_para, "PP_lstm_layer_num")
        self.PP_nheads = cnf.getint(PP_para, "PP_nheads")
        self.PP_dropout = cnf.getfloat(PP_para, "PP_dropout")
        self.PP_lr = cnf.getfloat(PP_para, "PP_lr")
        self.PP_batch_size = cnf.getint(PP_para, "PP_batch_size")
        self.PP_epoch = cnf.getint(PP_para, "PP_epoch")
        self.user_num = 0