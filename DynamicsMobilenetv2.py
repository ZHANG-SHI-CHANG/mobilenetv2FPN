from detection_dataloader import create_training_instances,BatchGenerator

import tensorflow as tf
from tensorflow.python.ops import array_ops

Detection_or_Classifier = 'classifier'#'detection','classifier'

class DynamicsMobilenetv2():
    MEAN = [103.94, 116.78, 123.68]
    NORMALIZER = 0.017
    
    def __init__(self,num_classes,num_anchors,batch_size,max_box_per_image,max_grid,ignore_thresh=0.6,learning_rate=0.045):
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.batch_size = batch_size
        self.max_box_per_image = max_box_per_image
        self.max_grid = max_grid
        self.ignore_thresh = ignore_thresh
        self.learning_rate = learning_rate
        
        self.__build()
    
    def __build(self):
        self.norm = 'batch_norm'#group_norm,batch_norm
        self.activate = 'prelu'#selu,leaky,swish,relu,relu6,prelu
        self.BlockInfo = {#scale /8
                          '1':[1,16,1,1,True],
                          '2':[6,24,1,2,True],
                          '3':[6,24,1,1,True],
                          '4':[6,32,1,2,True],
                          '5':[6,32,2+1,1,True],
                          '6':[6,64,1,2,True],
                          '7':[6,64,3+1,1,True],
                          '8':[6,96,1,1,True],
                          '9':[6,96,2+1,1,True],
                          '10':[6,160,1,2,True],
                          '11':[6,160,2+1,1,True],
                          '12':[6,320,1,1,True]}
    
        self.__init_global_epoch()
        self.__init_global_step()
        self.__init_input()
        
        with tf.variable_scope('zsc_preprocessing'):
            red, green, blue = tf.split(self.input_image, num_or_size_splits=3, axis=3)
            x = tf.concat([
                           tf.subtract(blue, DynamicsMobilenetv2.MEAN[0]) * DynamicsMobilenetv2.NORMALIZER,
                           tf.subtract(green, DynamicsMobilenetv2.MEAN[1]) * DynamicsMobilenetv2.NORMALIZER,
                           tf.subtract(red, DynamicsMobilenetv2.MEAN[2]) * DynamicsMobilenetv2.NORMALIZER,
                          ], 3)
            
        with tf.variable_scope('zsc_feature'):
            #none,none,none,3
            x = PrimaryConv('PrimaryConv',x,32,self.norm,self.activate,self.is_training)
            skip_0 = x
            #none,none/2,none/2,32
            
            index = '1'
            x = DynamicsMobilenetv2Block('DynamicsMobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_1 = x
            #none,none/2,none/2,16
            
            index = '2'
            x = DynamicsMobilenetv2Block('DynamicsMobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_2 = x
            #none,none/4,none/4,24
            
            index = '3'
            x = DynamicsMobilenetv2Block('DynamicsMobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_3 = x
            #none,none/4,none/4,24
            
            index = '4'
            x = DynamicsMobilenetv2Block('DynamicsMobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_4 = x
            #none,none/8,none/8,32
            
            index = '5'
            x = DynamicsMobilenetv2Block('DynamicsMobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_5 = x
            #none,none/8,none/8,32
            
            index = '6'
            x = DynamicsMobilenetv2Block('DynamicsMobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_6 = x
            #none,none/16,none/16,64
            
            index = '7'
            x = DynamicsMobilenetv2Block('DynamicsMobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_7 = x
            #none,none/16,none/16,64
            
            index = '8'
            x = DynamicsMobilenetv2Block('DynamicsMobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_8 = x
            #none,none/16,none/16,96
            
            index = '9'
            x = DynamicsMobilenetv2Block('DynamicsMobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_9 = x
            #none,none/16,none/16,96
            
            index = '10'
            x = DynamicsMobilenetv2Block('DynamicsMobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_10 = x
            #none,none/32,none/32,160
            
            index = '11'
            x = DynamicsMobilenetv2Block('DynamicsMobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_11 = x
            #none,none/32,none/32,160
            
            index = '12'
            x = DynamicsMobilenetv2Block('DynamicsMobilenetv2Block_'+index,x,
                                 self.BlockInfo[index][0],self.BlockInfo[index][1],self.BlockInfo[index][2],self.BlockInfo[index][3],self.BlockInfo[index][4],
                                 self.norm,self.activate,self.is_training)
            skip_12 = x
            #none,none/32,none/32,320
            
        if Detection_or_Classifier=='classifier':
            with tf.variable_scope('zsc_classifier'):
                global_pool = tf.reduce_mean(skip_12,[1,2],keep_dims=True)
                
                self.classifier_logits = tf.reshape(_conv_block('Logits',global_pool,self.num_classes,1,1,'SAME',self.norm,self.activate,self.is_training),
                                                    [tf.shape(global_pool)[0],self.num_classes])
        elif Detection_or_Classifier=='detection':
            with tf.variable_scope('zsc_detection'):
                with tf.variable_scope('zsc_conv_transpose'):
                    skip_3 = tf.concat([skip_12,skip_11,skip_10],axis=-1)
                    skip_2 = tf.concat([skip_9,skip_8,skip_7,skip_6],axis=-1)
                    skip_1 = tf.concat([skip_5,skip_4],axis=-1)
                    x = _conv_block('Conv_1',skip_3,skip_3.get_shape().as_list()[-1],1,1,'SAME',self.norm,self.activate,self.is_training)
                    pred1_x = x
                    
                    x = tf.image.resize_bilinear(x,[2*tf.shape(x)[1],2*tf.shape(x)[2]])
                    x = _conv_block('Conv_2',x,skip_2.get_shape().as_list()[-1]//2,1,1,'SAME',self.norm,self.activate,self.is_training)
                    
                    x = tf.concat([x,skip_2],axis=-1)
                    x = _conv_block('Conv_3',x,skip_2.get_shape().as_list()[-1]//2,1,1,'SAME',self.norm,self.activate,self.is_training)
                    pred2_x = x
                    
                    x = tf.image.resize_bilinear(x,[2*tf.shape(x)[1],2*tf.shape(x)[2]])
                    x = _conv_block('Conv_4',x,skip_1.get_shape().as_list()[-1]//2,1,1,'SAME',self.norm,self.activate,self.is_training)
                    
                    x = tf.concat([x,skip_1],axis=-1)
                    x = _conv_block('Conv_5',x,skip_1.get_shape().as_list()[-1]//2,1,1,'SAME',self.norm,self.activate,self.is_training)
                    pred3_x = x
                    
                with tf.variable_scope('zsc_pred'):
                    pred1_x = _conv_block('Conv_1_1',pred1_x,self.num_anchors*(5+self.num_classes),1,1,'SAME',self.norm,self.activate,self.is_training)
                    pred1_x = _conv_block('Conv_1_2',pred1_x,self.num_anchors*(5+self.num_classes),3,1,'SAME',self.norm,self.activate,self.is_training)
                    self.pred_yolo_1 = Attention('Attention_1',pred1_x,self.norm,self.activate,self.is_training)
                    #none,none/32,none/32,5*(5+self.num_classes)
                    
                    pred2_x = _conv_block('Conv_2_1',pred2_x,self.num_anchors*(5+self.num_classes),1,1,'SAME',self.norm,self.activate,self.is_training)
                    pred2_x = _conv_block('Conv_2_2',pred2_x,self.num_anchors*(5+self.num_classes),3,1,'SAME',self.norm,self.activate,self.is_training)
                    self.pred_yolo_2 = Attention('Attention_2',pred2_x,self.norm,self.activate,self.is_training)
                    #none,none/16,none/16,5*(5+self.num_classes)
                    
                    pred3_x = _conv_block('Conv_3_1',pred3_x,self.num_anchors*(5+self.num_classes),1,1,'SAME',self.norm,self.activate,self.is_training)
                    pred3_x = _conv_block('Conv_3_2',pred3_x,self.num_anchors*(5+self.num_classes),3,1,'SAME',self.norm,self.activate,self.is_training)
                    self.pred_yolo_3 = Attention('Attention_3',pred3_x,self.norm,self.activate,self.is_training)
                    #none,none/8,none/8,5*(5+self.num_classes)
        self.__init__output()
        
        if Detection_or_Classifier=='classifier':
            pass
        elif Detection_or_Classifier=='detection':
            self.__prob()
        
    def __prob(self):
        def decode_pred(pred,anchors,net_factor,obj_thresh=0.6):
            pred_list = tf.split(tf.expand_dims(pred,axis=3),num_or_size_splits=self.num_anchors,axis=4)#[(none,none/n,none/n,1,C//3),...]
            pred = tf.concat(pred_list,axis=3)#none,none/n,none/n,3,C//3
            
            max_grid_h, max_grid_w = self.max_grid
            cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
            cell_y = tf.transpose(cell_x, (0,2,1,3,4))
            cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [1, 1, 1, self.num_anchors, 1])#(none,max_grid_h,max_grid_w,3,2)

            grid_h = tf.shape(pred)[1]
            grid_w = tf.shape(pred)[2]
            grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])
            
            anchors = tf.reshape(tf.constant(anchors,dtype=tf.float32),[1,1,1,-1,2])*net_factor/self.original_wh#1,1,1,3,2
            #################################################################################################
            
            pred_xy = ((cell_grid[:,:grid_h,:grid_w,:,:] + tf.sigmoid(pred[...,:2]))/grid_factor)#none,grid_h,grid_w,3,2 每个网格内偏移后的框并归一化
            pred_wh = anchors*tf.exp(pred[...,2:4])/net_factor#none,grid_h,grid_w,3,2 每个网格的宽高
            pred_min_xy = pred_xy-pred_wh/2.0
            pred_max_xy = pred_xy+pred_wh/2.0
            pred_conf = tf.expand_dims(tf.sigmoid(pred[...,4]),axis=-1)#none,grid_h,grid_w,3,1
            pred_class = pred_conf*tf.sigmoid(pred[...,5:])#none,grid_h,grid_w,3,self.num_classes
            pred_class *= tf.cast(pred_class>obj_thresh,tf.float32)#none,grid_h,grid_w,3,self.num_classes
            pred = tf.concat([pred_min_xy,pred_max_xy,pred_conf,pred_class],axis=-1)#none,grid_h,grid_w,3,4+1+self.num_classes
            
            #!!目前这个程序只能对单张图像进行测试
            obj_index = tf.where(pred[...,4]>obj_thresh)#框数,4   置信索引
            obj_index_mask = tf.cast(obj_index[:,0],tf.int32)#把第一维batch维取出来,作为索引掩膜
            #先把索引掩膜去重编号,然后计算每个编号的数量,即计算每个batch的框的数量(考虑用tf.bincount比较合适因为可能有batch不存在框,不存在的batch返回计数0)
            #infos_split_group = tf.unique_with_counts(tf.unique(obj_index_mask)[1])[2]
            #infos_split_group = tf.bincount(obj_index_mask)
            
            infos = tf.gather_nd(pred,obj_index)#框数,4+1+self.num_classes
            
            return infos
        def correct_boxes(infos):
            pred_min_xy = infos[:,:2]*self.original_wh#框数,2
            pred_max_xy = infos[:,2:4]*self.original_wh#框数,2
            
            zeros = tf.zeros_like(pred_min_xy)
            ones = self.original_wh*tf.ones_like(pred_min_xy)
            pred_min_xy = tf.where(pred_min_xy>zeros,pred_min_xy,zeros)
            pred_min_xy = tf.where(pred_min_xy<ones,pred_min_xy,ones)
            pred_max_xy = tf.where(pred_max_xy>zeros,pred_max_xy,zeros)
            pred_max_xy = tf.where(pred_max_xy<ones,pred_max_xy,ones)
            return tf.concat([pred_min_xy,pred_max_xy,infos[:,4:]],axis=-1)
        def nms(infos,nms_threshold=0.4):
            #提取batch
            #infos_mask = tf.ones_like(infos)#框数,4+1+self.num_classes
            #batch = tf.cast(tf.reduce_sum(infos_mask)/tf.reduce_sum(infos_mask,axis=1)[0],tf.int32)
            batch = tf.shape(infos)[0]
            
            #先把infos按照最大class概率重排序
            #pred_max_class = tf.reduce_max(infos[:,5:],axis=1)#batch,
            #ix = tf.nn.top_k(tf.transpose(tf.expand_dims(pred_max_class,axis=1),[1,0]), batch, sorted=True, name="top_anchors").indices#1,batch
            #infos = tf.gather_nd(infos,tf.transpose(ix,[1,0]))#batch,4+1+self.num_classes
            
            pred_min_yx = infos[:,1::-1]#batch,2
            pred_max_yx = infos[:,3:1:-1]#batch,2
            pred_yx = tf.concat([pred_min_yx,pred_max_yx],axis=-1)#batch,4
            pred_max_class = tf.reduce_max(infos[:,5:],axis=1)#batch,
            indices = tf.image.non_max_suppression(pred_yx, pred_max_class, batch,nms_threshold, name="non_max_suppression")
            infos = tf.gather(infos, indices,name='zsc_output')
            
            return infos
        
        anchors = [[190,212, 245,348, 321,150, 343,256, 372,379],[84,86, 108,162, 109,288, 162,329, 174,103],[18,27, 28,75, 49,132, 55,43, 65,227]]
        
        #input_mask = tf.ones_like(self.input_image)#none,none,none,3
        #net_h = tf.cast(tf.reduce_sum(input_mask)/tf.reduce_sum(input_mask,axis=[0,2,3])[0],tf.int32)#net_h = none
        #net_w = tf.cast(tf.reduce_sum(input_mask)/tf.reduce_sum(input_mask,axis=[0,1,3])[0],tf.int32)#net_w = none
        #net_factor = tf.reshape(tf.cast([net_w,net_h],tf.float32),[1,1,1,1,2])#1,1,1,1,2
        net_h = tf.shape(self.input_image)[1]
        net_w = tf.shape(self.input_image)[2]
        net_factor = tf.reshape(tf.cast([net_w,net_h],tf.float32),[1,1,1,1,2])
        
        infos_1 = decode_pred(self.pred_yolo_1,anchors[0],net_factor)
        infos_2 = decode_pred(self.pred_yolo_2,anchors[0],net_factor)
        infos_3 = decode_pred(self.pred_yolo_3,anchors[0],net_factor)
        infos = tf.concat([infos_1,infos_2,infos_3],axis=0)#框数,4+1+self.num_classes
        
        infos = correct_boxes(infos)#框数,4+1+self.num_classes
        
        self.infos = nms(infos)#框数,4+1+self.num_classes
    def __init__output(self):
        with tf.variable_scope('output'):
            regularzation_loss = sum(tf.get_collection("regularzation_loss"))
            
            if Detection_or_Classifier=='classifier':
                self.all_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.classifier_logits, labels=self.y, name='loss'))
                self.all_loss += regularzation_loss
                
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    learning_rate = tf.train.exponential_decay(self.learning_rate,global_step=self.global_epoch_tensor,decay_steps=5,decay_rate=0.98)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate)
                    self.train_op = self.optimizer.minimize(self.all_loss,global_step=self.global_epoch_tensor)
                
                self.y_out_softmax = tf.nn.softmax(self.classifier_logits,name='zsc_output')
                
                self.y_out_argmax = tf.cast(tf.argmax(self.y_out_softmax, axis=-1),tf.int32)
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_out_argmax), tf.float32))
                
                self.accuracy_top_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.y_out_softmax,self.y,5),tf.float32))

                with tf.name_scope('train-summary-per-iteration'):
                    tf.summary.scalar('loss', self.all_loss)
                    tf.summary.scalar('acc', self.accuracy)
                    tf.summary.scalar('acc', self.accuracy_top_5)
                    self.summaries_merged = tf.summary.merge_all()
            elif Detection_or_Classifier=='detection':
                self.loss_yolo_1,class_loss_1,recall50_1,recall75_1,class_acc_1,avg_obj_1,avg_noobj_1,count_1,count_noobj_1 = self.loss(1,[self.input_image, self.pred_yolo_1, self.true_yolo_1, self.true_boxes],
                                              self.anchors[:,20:], [1*num for num in self.max_grid], self.ignore_thresh)
                self.loss_yolo_2,class_loss_2,recall50_2,recall75_2,class_acc_2,avg_obj_2,avg_noobj_2,count_2,count_noobj_2 = self.loss(2,[self.input_image, self.pred_yolo_2, self.true_yolo_2, self.true_boxes],
                                              self.anchors[:,10:20], [1*num for num in self.max_grid], self.ignore_thresh)
                self.loss_yolo_3,class_loss_3,recall50_3,recall75_3,class_acc_3,avg_obj_3,avg_noobj_3,count_3,count_noobj_3 = self.loss(4,[self.input_image, self.pred_yolo_3, self.true_yolo_3, self.true_boxes],
                                              self.anchors[:,:10], [1*num for num in self.max_grid], self.ignore_thresh)
                self.all_loss = tf.reduce_mean(tf.sqrt(self.loss_yolo_1+self.loss_yolo_2+self.loss_yolo_3)+class_loss_1+class_loss_2+class_loss_3) + regularzation_loss
                
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    learning_rate = tf.train.exponential_decay(self.learning_rate,global_step=self.global_epoch_tensor,decay_steps=5,decay_rate=0.98)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate)
                    self.train_op = self.optimizer.minimize(self.all_loss,global_step=self.global_epoch_tensor)
                
                with tf.name_scope('train-summary-per-iteration'):
                    tf.summary.scalar('loss', tf.cast(self.all_loss,tf.float32))
                    tf.summary.scalar('recall50',(recall50_1*count_1+recall50_2*count_2+recall50_3*count_3)/(count_1+count_2+count_3+1e-3) )
                    tf.summary.scalar('recall75',(recall75_1*count_1+recall75_2*count_2+recall75_3*count_3)/(count_1+count_2+count_3+1e-3))
                    tf.summary.scalar('class_accury',(class_acc_1*count_1+class_acc_2*count_2+class_acc_3*count_3)/(count_1+count_2+count_3+1e-3))
                    tf.summary.scalar('avg obj accuracy',(avg_obj_1*count_1+avg_obj_2*count_2+avg_obj_3*count_3)/(count_1+count_2+count_3+1e-3))
                    tf.summary.scalar('avg no obj accuracy',(avg_noobj_1*count_1+avg_noobj_2*count_2+avg_noobj_3*count_3)/(count_1+count_2+count_3+1e-3))
                    tf.summary.scalar('obj count',(count_1+count_2+count_3))
                    tf.summary.scalar('no obj count',(count_noobj_2+count_noobj_2+count_noobj_3))
                    self.summaries_merged = tf.summary.merge_all()
    def loss(self,scale,x,anchors, max_grid,ignore_thresh):
        def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
            sigmoid_p = tf.nn.sigmoid(prediction_tensor)
            zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
            
            pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
            
            neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
            per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                                  - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
            return per_entry_cross_ent
        max_grid_h, max_grid_w = max_grid
        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
        cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [self.batch_size, 1, 1, self.num_anchors, 1])#max_grid的索引grid  (1,max_grid_h,max_grid_w,5,1)
        
        input_image, y_pred, y_true, true_boxes = x#(none,H,W,3),(none,grid_h,grid_w,3*(4+1+num_classes))
                                                                #(none,grid_h,grid_w,3,4+1+num_classes),(none,1,1,1,max_box_per_image,4)

        # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+num_classes]
        y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([self.num_anchors, -1])], axis=0))#(none,grid_h,grid_w,3,4+1+num_classes)
        
        # initialize the masks
        object_mask     = tf.expand_dims(y_true[..., 4], 4)#score (none,grid_h,grid_w,3,1)  y_true里面每个groundtruth只匹配一个特征层的一个anchor，概率为1
        no_object_mask  = 1 - object_mask                  #      (none,grid_h,grid_w,3,1)

        # the variable to keep track of number of batches processed
        batch_seen = tf.Variable(0.)
        
        # compute grid factor and net factor
        grid_h      = tf.shape(y_true)[1]
        grid_w      = tf.shape(y_true)[2]
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])#loss输入层grid (1,1,1,1,2)
        
        net_h       = tf.shape(input_image)[1]
        net_w       = tf.shape(input_image)[2]            
        net_factor  = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1,1,1,1,2])#输入图像grid     (1,1,1,1,2)
        
        anchors = tf.reshape(anchors,[-1,1,1,self.num_anchors,2])#none,1,1,3,2
        _anchors = anchors/net_factor#none,1,1,3,2
        anchors_min_wh = tf.where(_anchors[:,:,:,0,0]<_anchors[:,:,:,0,1],_anchors[:,:,:,0,0],_anchors[:,:,:,0,1])#none,1,1
        anchors_min_wh = tf.expand_dims(anchors_min_wh,-1)#none,1,1,1
        anchors_max_wh = tf.where(_anchors[:,:,:,-1,0]>_anchors[:,:,:,-1,1],_anchors[:,:,:,-1,0],_anchors[:,:,:,-1,1])#none,1,1
        anchors_max_wh = tf.expand_dims(anchors_max_wh,-1)#none,1,1,1
        anchors = tf.tile(anchors,[1,grid_h,grid_w,1,1])#none.grid_h,grid_w,3,2
        
        """
        Adjust prediction
        """
        pred_box_xy    = (cell_grid[:,:grid_h,:grid_w,:,:] + tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy          (none,grid_h,grid_w,3,2)
        pred_box_wh    = y_pred[..., 2:4]                                                       # t_wh                        (none,grid_h,grid_w,3,2)
        pred_box_conf  = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)                          # adjust confidence           (none,grid_h,grid_w,3,1)
        pred_box_class = y_pred[..., 5:]                         # adjust class probabilities  (none,grid_h,grid_w,3,num_classes)
        
        """
        Adjust ground truth
        """
        true_box_xy    = y_true[..., 0:2] # (sigma(t_xy) + c_xy)  (none,grid_h,grid_w,3,2)
        true_box_wh    = y_true[..., 2:4] # t_wh                  (none,grid_h,grid_w,3,2)
        true_box_conf  = tf.expand_dims(y_true[..., 4], 4)#       (none,grid_h,grid_w,3,1)
        true_box_class = tf.argmax(y_true[..., 5:], -1)   #       (none,grid_h,grid_w,3)

        """
        Compare each predicted box to all true boxes
        """        
        # initially, drag all objectness of all boxes to 0
        conf_delta  = pred_box_conf - 0                          #(none,grid_h,grid_w,3,1)

        # then, ignore the boxes which have good overlap with some true box
        true_xy = true_boxes[..., 0:2] / grid_factor             #(none,1,1,1,max_box_per_image,2)/(1,1,1,1,2)=(none,1,1,1,max_box_per_image,2)
        true_wh = true_boxes[..., 2:4] / net_factor              #(none,1,1,1,max_box_per_image,2)/(1,1,1,1,2)=(none,1,1,1,max_box_per_image,2)
        '''
        ###################
        ###################
        #snip
        true_min_wh = tf.minimum(true_wh[...,0],true_wh[...,1])#none,1,1,1,max_box_per_image
        snip_mask = tf.cast(true_min_wh>tf.expand_dims(anchors_min_wh,4)*2/3,
                            tf.float32)*tf.cast(true_min_wh<tf.expand_dims(anchors_max_wh,4)*1.5,tf.float32)#none,1,1,1,max_box_per_image
        snip_mask = tf.expand_dims(snip_mask,-1)#none,1,1,1,max_box_per_image,1
        true_wh = snip_mask*true_wh#none,1,1,1,max_box_per_image,2
        true_xy = snip_mask*true_xy#none,1,1,1,max_box_per_image,2
        ###################
        ###################
        '''
        #在输入feature map尺度上groundtruth的标记框
        true_wh_half = true_wh / 2.                              #(none,1,1,1,max_box_per_image,2)
        true_mins    = true_xy - true_wh_half                    #(none,1,1,1,max_box_per_image,2)
        true_maxes   = true_xy + true_wh_half                    #(none,1,1,1,max_box_per_image,2)
        
        pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)                       #(none,grid_h,grid_w,3,1,2)
        pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * anchors / net_factor, 4) #(none,grid_h,grid_w,3,1,2)
        
        #feature map预测的标记框
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half                     #(none,grid_h,grid_w,3,1,2)
        pred_maxes   = pred_xy + pred_wh_half                     #(none,grid_h,grid_w,3,1,2)
        
        #计算feature map预测和groundtruth的所有iou
        intersect_mins  = tf.maximum(pred_mins,  true_mins)                  #(none,grid_h,grid_w,3,max_box_per_image,2)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)                 #(none,grid_h,grid_w,3,max_box_per_image,2)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)   #(none,grid_h,grid_w,3,max_box_per_image,2)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]        #(none,grid_h,grid_w,3,max_box_per_image)
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]                       #(none,1,1,1,max_box_per_image)
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]                       #(none,grid_h,grid_w,3,1)

        union_areas = pred_areas + true_areas - intersect_areas              #(none,grid_h,grid_w,3,max_box_per_image)
        iou_scores  = tf.truediv(intersect_areas, union_areas)               #(none,grid_h,grid_w,3,max_box_per_image)

        #计算feature map每个位置匹配到的3个最大iou，大于阈值的conf_delta置0
        best_ious   = tf.reduce_max(iou_scores, axis=4)                              #(none,grid_h,grid_w,3)
        conf_delta *= tf.expand_dims(tf.to_float(best_ious < ignore_thresh), 4) #(none,grid_h,grid_w,3,1)

        """
        Compute some online statistics
        """            
        true_xy = true_box_xy / grid_factor                                   #(none,grid_h,grid_w,3,2)
        true_wh = tf.exp(true_box_wh) * anchors / net_factor             #(none,grid_h,grid_w,3,2)
        '''
        ###################
        ###################
        #snip
        true_min_wh = tf.minimum(true_wh[...,0],true_wh[...,1])#none,grid_h,grid_w,3
        anchors_min_wh = tf.tile(anchors_min_wh,[1,grid_h,grid_w,self.num_anchors])
        anchors_max_wh = tf.tile(anchors_max_wh,[1,grid_h,grid_w,self.num_anchors])
        snip_mask = tf.cast(true_min_wh>anchors_min_wh*2/3,
                            tf.float32)*tf.cast(true_min_wh<anchors_max_wh*1.5,tf.float32)#none,grid_h,grid_w,3
        snip_mask = tf.expand_dims(snip_mask,-1)#none,grid_h,grid_w,3,1
        true_xy = snip_mask*true_xy#none,grid_h,grid_w,3,2
        true_wh = snip_mask*true_wh#none,grid_h,grid_w,3,2
        true_box_xy = snip_mask*true_box_xy#none,grid_h,grid_w,3,2
        true_box_wh = snip_mask*true_box_wh#none,grid_h,grid_w,3,2
        true_box_conf = snip_mask*true_box_conf#none,grid_h,grid_w,3,1
        true_box_class = tf.cast(tf.squeeze(snip_mask,-1),tf.int64)*true_box_class#none,grid_h,grid_w,3
        
        pred_box_xy = snip_mask*pred_box_xy#none,grid_h,grid_w,3,2
        pred_box_wh = snip_mask*pred_box_wh#none,grid_h,grid_w,3,2
        pred_box_conf = snip_mask*pred_box_conf#none,grid_h,grid_w,3,1
        pred_box_class = snip_mask*pred_box_class#none,grid_h,grid_w,3,2
        
        object_mask = snip_mask*object_mask#none,grid_h,grid_w,3,1
        ###################
        ###################
        '''
        #在输入feature map尺度上为每个groundtruth匹配到的惟一的标记框
        true_wh_half = true_wh / 2.                                           #(none,grid_h,grid_w,3,2)
        true_mins    = true_xy - true_wh_half                                 #(none,grid_h,grid_w,3,2)
        true_maxes   = true_xy + true_wh_half                                 #(none,grid_h,grid_w,3,2)

        pred_mins    = tf.squeeze(pred_mins,axis=4)                           #(none,grid_h,grid_w,3,2)
        pred_maxes   = tf.squeeze(pred_maxes,axis=4)                          #(none,grid_h,grid_w,3,2)
        
        #计算feature map预测和每个groundtruth匹配到的唯一标记框的iou
        intersect_mins  = tf.maximum(pred_mins,  true_mins)                   #(none,grid_h,grid_w,3,2)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)                  #(none,grid_h,grid_w,3,2)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)    #(none,grid_h,grid_w,3,2)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]         #(none,grid_h,grid_w,3)
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]                        #(none,grid_h,grid_w,3)
        pred_areas = tf.squeeze(pred_areas,axis=4)                            #(none,grid_h,grid_w,3)

        union_areas = pred_areas + true_areas - intersect_areas               #(none,grid_h,grid_w,3)
        iou_scores  = tf.truediv(intersect_areas, union_areas)                #(none,grid_h,grid_w,3)
        
        #计算feature map预测里面每个groundtruth匹配到的唯一标记框的iou分数，其他预测框分数置0
        iou_scores  = object_mask * tf.expand_dims(iou_scores, 4)             #(none,grid_h,grid_w,3,1)

        count       = tf.reduce_sum(tf.to_float(object_mask))                                       #(none,)  计算唯一标记框的数量
        count_noobj = tf.reduce_sum(tf.to_float(no_object_mask))                                    #(none,)  计算非标记框的数量
        detect_mask = tf.to_float((pred_box_conf*object_mask) >= 0.5)
        class_mask  = tf.expand_dims(tf.to_float(tf.equal(tf.argmax(pred_box_class, -1), true_box_class)), 4)
        recall50    = tf.reduce_sum(tf.to_float(iou_scores >= 0.5 ) * detect_mask  * class_mask) / (count + 1e-3)#(none,)  计算唯一标记框里iou>0.5的分数 recall50
        recall75    = tf.reduce_sum(tf.to_float(iou_scores >= 0.75) * detect_mask  * class_mask) / (count + 1e-3)#(none,)  计算唯一标记框里iou>0.75的分数 recall75
        avg_iou     = tf.reduce_sum(iou_scores) / (count + 1e-3)                                    #(none,)  计算唯一标记框的平均iou
        avg_obj     = tf.reduce_sum(object_mask * pred_box_conf)  / (count + 1e-3) #(none,)  计算唯一标记框的预测平均置信度
        avg_noobj   = tf.reduce_sum(no_object_mask * pred_box_conf)  / (count_noobj + 1e-3)         #(none,)  计算非标记框里的预测平均置信度
        avg_cat     = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3) #(none,)  计算唯一标记框的类别平均预测准确率

        """
        Warm-up training
        """
        batch_seen = tf.assign_add(batch_seen, 1.)
        true_box_xy, true_box_wh, xywh_mask = true_box_xy, true_box_wh, object_mask
        """
        Compare each true box to all anchor boxes
        """
        xywh_scale = tf.exp(true_box_wh) * anchors / net_factor                     #(none,grid_h,grid_w,3,2)
        xywh_scale = tf.expand_dims(2 - xywh_scale[..., 0] * xywh_scale[..., 1], axis=4) #(none,grid_h,grid_w,3,1) 真值框尺寸越小，scale越大

        xy_delta    = scale*5*xywh_mask   * (pred_box_xy-true_box_xy) * xywh_scale                          #(none,grid_h,grid_w,3,2) 计算唯一标记框的预测和真值xy偏差
        wh_delta    = scale*5*xywh_mask   * (pred_box_wh-true_box_wh) * xywh_scale                          #(none,grid_h,grid_w,3,2) 计算唯一标记框的预测和真值wh偏差
        conf_delta_obj = scale*5*object_mask*(pred_box_conf-true_box_conf)#(none,grid_h,grid_w,3,1) 计算唯一预测框的置信度偏差
        conf_delta_noobj = 1*no_object_mask*conf_delta*(1.3-1.0*count_noobj/(count+count_noobj))#(none,grid_h,grid_w,3,1) 计算非唯一预测框的置信度偏差
        class_delta = scale*5*object_mask * tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class), 4)
                                                      #(none,grid_h,grid_w,3,num_classes) 计算唯一标记框的分类偏差
        #conf_delta_obj = focal_loss(tf.reshape(pred_box_conf,[tf.shape(pred_box_conf)[0],-1,1]),tf.reshape(true_box_conf,[tf.shape(true_box_conf)[0],-1,1]))
        #conf_delta_obj = 5*object_mask*tf.reshape(conf_delta_obj,tf.concat([tf.shape(pred_box_conf)[:-1],[-1]],axis=-1))
        #conf_delta_noobj = focal_loss(tf.reshape(conf_delta,[tf.shape(conf_delta)[0],-1,1]),tf.reshape(tf.ones_like(conf_delta),[tf.shape(conf_delta)[0],-1,1]))
        #conf_delta_noobj = 1*no_object_mask*tf.reshape(conf_delta_noobj,tf.concat([tf.shape(conf_delta)[:-1],[-1]],axis=-1))
        #class_delta = focal_loss(tf.reshape(pred_box_class,[tf.shape(pred_box_class)[0],-1,self.num_classes]),
        #                         tf.reshape(tf.one_hot(true_box_class,self.num_classes),[tf.shape(pred_box_class)[0],-1,self.num_classes]))
        #class_delta = 1*object_mask*tf.reshape(class_delta,tf.concat([tf.shape(pred_box_class)[:-1],[-1]],axis=-1))
        
        loss = tf.reduce_sum(tf.square(xy_delta),       list(range(1,5))) + \
               tf.reduce_sum(tf.square(wh_delta),       list(range(1,5))) + \
               tf.reduce_sum(tf.square(conf_delta_obj),     list(range(1,5))) + \
               tf.reduce_sum(tf.square(conf_delta_noobj),     list(range(1,5)))
        class_loss = tf.reduce_sum(class_delta,    list(range(1,5)))#(none,)
        
        loss = tf.Print(loss, [grid_h, count], message='obj num: ', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_obj], message='obj average confidence: ', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_noobj], message='unobj average confidence: ', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_iou], message='obj average iou: ', summarize=1000)
        loss = tf.Print(loss, [grid_h, avg_cat], message='obj average class average accuracy: ', summarize=1000)
        loss = tf.Print(loss, [grid_h, recall50], message='obj recall50: ', summarize=1000)
        loss = tf.Print(loss, [grid_h, recall75], message='obj recall75: ', summarize=1000)
        loss = tf.Print(loss, [grid_h, tf.reduce_sum(tf.square(xy_delta)),tf.reduce_sum(tf.square(wh_delta)),
                                       tf.reduce_sum(tf.square(conf_delta_obj)),tf.reduce_sum(tf.square(conf_delta_noobj)),
                                       tf.reduce_sum(class_delta)],  message='xy loss,wh loss,conf loss,un conf loss,class loss:\n',   summarize=1000)

        return loss,class_loss,recall50,recall75,avg_cat,avg_obj,avg_noobj,count,count_noobj
    def __init_input(self):
        if Detection_or_Classifier=='classifier':
            with tf.variable_scope('input'):
                self.input_image = tf.placeholder(tf.float32,[None, None, None, 3],name='zsc_input')#训练、测试用
                self.y = tf.placeholder(tf.int32, [None],name='zsc_input_target')#训练、测试用
                self.is_training = tf.placeholder(tf.float32,name='zsc_is_train')#训练、测试用
                self.is_training = tf.equal(self.is_training,1.0)
        elif Detection_or_Classifier=='detection':
            with tf.variable_scope('input'):
                self.input_image = tf.placeholder(tf.float32,[None,None,None,3],name='zsc_input')#训练、测试用
                self.original_wh = tf.placeholder(tf.float32,[None,2],name='zsc_original_wh')#仅测试用
                self.is_training = tf.placeholder(tf.float32,name='zsc_is_train')#训练、测试（不一定）用
                self.is_training = tf.equal(self.is_training,1.0)
                self.anchors = tf.placeholder(tf.float32,[None,self.num_anchors*3*2])#训练用
                self.true_boxes = tf.placeholder(tf.float32,[None,1, 1, 1, self.max_box_per_image, 4])#训练用
                self.true_yolo_1 = tf.placeholder(tf.float32,[None,None, None, self.num_anchors, 4+1+self.num_classes])#训练用
                self.true_yolo_2 = tf.placeholder(tf.float32,[None,None, None, self.num_anchors, 4+1+self.num_classes])#训练用
                self.true_yolo_3 = tf.placeholder(tf.float32,[None,None, None, self.num_anchors, 4+1+self.num_classes])#训练用
    def __init_global_epoch(self):
        with tf.variable_scope('global_epoch'):
            self.global_epoch_tensor = tf.Variable(-1, trainable=False, name='global_epoch')
            self.global_epoch_input = tf.placeholder('int32', None, name='global_epoch_input')
            self.global_epoch_assign_op = self.global_epoch_tensor.assign(self.global_epoch_input)
    def __init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
            self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

################################################################################################################
################################################################################################################
################################################################################################################
##Mobilenetv2Block
def DynamicsMobilenetv2Block(name,x,ratio=6,num_filters=16,repeat=1,stride=1,use_depthwise=True,norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        if stride==1:
            FeatureList = [x]
            x = DepthwiseBlock('DepthwiseBlock_0',x,ratio,num_filters,1.0,use_depthwise,norm,activate,is_training)
            FeatureList.append(x)
            
            if repeat>1:
                with tf.variable_scope('loop'):
                    x0 = FeatureList[0]
                    x = FeatureList[1]
                    
                    identityWeight = tf.get_variable('k',[1],tf.float32,initializer=tf.random_uniform_initializer(-0.1,0))
                    x = DepthwiseBlock('DepthwiseBlock_1',x,ratio,num_filters,1.0-identityWeight,use_depthwise,norm,activate,is_training)
                    if x0.get_shape().as_list()[-1]==x.get_shape().as_list()[-1]:
                        pass
                    else:
                        x0 = _B_conv_block('conv',x0,num_filters,1,1,'SAME',norm,activate,is_training)
                    x += identityWeight*x0
                    
                    del FeatureList[0]
                    FeatureList.append(x)
                
                for _ in range(2,repeat):
                    with tf.variable_scope('loop',reuse=True):
                        x0 = FeatureList[0]
                        x = FeatureList[1]
                        
                        identityWeight = tf.get_variable('k',[1],tf.float32,initializer=tf.random_uniform_initializer(-0.1,0))
                        x = DepthwiseBlock('DepthwiseBlock_1',x,ratio,num_filters,1.0-identityWeight,use_depthwise,norm,activate,is_training)
                        x += identityWeight*x0
                        
                        del FeatureList[0]
                        FeatureList.append(x)
        else:
            x = Transition('Transition',x,ratio,num_filters,use_depthwise,norm,activate,is_training)
        
        return x
def DepthwiseBlock(name,x,ratio,num_filters=16,identityWeight=1.0,use_depthwise=True,norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        if use_depthwise:
            input = x
            
            x = _B_conv_block('conv_0',x,ratio*num_filters,1,1,'SAME',norm,activate,is_training)
            x = _B_depthwise_conv2d('depthwise',x,1,3,1,'SAME',norm,activate,is_training)
            x = _B_conv_block('conv_1',x,num_filters,1,1,'SAME',norm,activate,is_training)
            
            x = x*SE('SE',x,norm,activate,is_training)
            
            if input.get_shape().as_list()[-1]==x.get_shape().as_list()[-1]:
                pass
            else:
                input = _B_conv_block('conv_2',input,num_filters,1,1,'SAME',norm,activate,is_training)
            x += input*identityWeight
        else:
            x = _B_conv_block('conv',x,num_filters,1,1,'SAME',norm,activate,is_training)
        return x
def Transition(name,x,ratio=6,num_filters=16,use_depthwise=True,norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        if use_depthwise:
            x = _B_conv_block('conv_0',x,ratio*num_filters,1,1,'SAME',norm,activate,is_training)
            x = _B_depthwise_conv2d('depthwise',x,1,3,2,'SAME',norm,activate,is_training)
            x = _B_conv_block('conv_1',x,num_filters,1,1,'SAME',norm,activate,is_training)
        else:
            x = _conv_block('conv',x,num_filters,3,2,'SAME',norm,activate,is_training)
        
        x = Attention('Attention',x,norm,activate,is_training)
        return x
##primary_conv
def PrimaryConv(name,x,num_filters=32,norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        #none,none,none,3
        x = _conv_block('conv_0',x,num_filters,3,2,'SAME',norm,activate,is_training)#none,none/2,none/2,num_filters
        return x
##_B_conv_block
##_conv_block
def _B_conv_block(name,x,num_filters=16,kernel_size=3,stride=2,padding='SAME',norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        if norm=='batch_norm':
            x = tf.layers.batch_normalization(x, training=is_training, epsilon=0.001,name='batchnorm')
        elif norm=='group_norm':
            x = group_norm(x,name='groupnorm')
        else:
            b = tf.get_variable('bias',num_filters,tf.float32,initializer=tf.constant_initializer(0.001))
            x += b
        if activate=='leaky': 
            x = LeakyRelu(x,leak=0.1, name='leaky')
        elif activate=='selu':
            x = selu(x,name='selu')
        elif activate=='swish':
            x = swish(x,name='swish')
        elif activate=='relu':
            x = tf.nn.relu(x,name='relu')
        elif activate=='relu6':
            x = tf.nn.relu6(x,name='relu6')
        elif activate=='prelu':
            x = prelu(x)
        else:
            pass
        
        w = GetWeight('weight',[kernel_size,kernel_size,x.shape.as_list()[-1],num_filters])
        x = tf.nn.conv2d(x,w,[1,stride,stride,1],padding=padding,name='conv')

        return x
##_conv_block
def _conv_block(name,x,num_filters=16,kernel_size=3,stride=2,padding='SAME',norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        w = GetWeight('weight',[kernel_size,kernel_size,x.shape.as_list()[-1],num_filters])
        x = tf.nn.conv2d(x,w,[1,stride,stride,1],padding=padding,name='conv')
        
        if norm=='batch_norm':
            x = tf.layers.batch_normalization(x, training=is_training, epsilon=0.001,name='batchnorm')
        elif norm=='group_norm':
            x = group_norm(x,name='groupnorm')
        else:
            b = tf.get_variable('bias',num_filters,tf.float32,initializer=tf.constant_initializer(0.001))
            x += b
        if activate=='leaky': 
            x = LeakyRelu(x,leak=0.1, name='leaky')
        elif activate=='selu':
            x = selu(x,name='selu')
        elif activate=='swish':
            x = swish(x,name='swish')
        elif activate=='relu':
            x = tf.nn.relu(x,name='relu')
        elif activate=='relu6':
            x = tf.nn.relu6(x,name='relu6')
        elif activate=='prelu':
            x = prelu(x)
        else:
            pass

        return x
##_B_depthwise_conv2d
def _B_depthwise_conv2d(name,x,scale=1,kernel_size=3,stride=1,padding='SAME',norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name) as scope:
        if norm=='batch_norm':
            x = tf.layers.batch_normalization(x, training=is_training, epsilon=0.001,name='batchnorm')
        elif norm=='group_norm':
            x = group_norm(x,name='groupnorm')
        else:
            b = tf.get_variable('bias',scale,tf.float32,initializer=tf.constant_initializer(0.001))
            x += b
        if activate=='leaky': 
            x = LeakyRelu(x,leak=0.1, name='leaky')
        elif activate=='selu':
            x = selu(x,name='selu')
        elif activate=='swish':
            x = swish(x,name='swish')
        elif activate=='relu':
            x = tf.nn.relu(x,name='relu')
        elif activate=='relu6':
            x = tf.nn.relu6(x,name='relu6')
        elif activate=='prelu':
            x = prelu(x)
        else:
            pass
        
        w = GetWeight('weight',[kernel_size,kernel_size,x.shape.as_list()[-1],scale])
        x = tf.nn.depthwise_conv2d(x, w, [1,stride,stride,1], padding)
        
        return x
##_depthwise_conv2d
def _depthwise_conv2d(name,x,scale=1,kernel_size=3,stride=1,padding='SAME',norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name) as scope:
        w = GetWeight('weight',[kernel_size,kernel_size,x.shape.as_list()[-1],scale])
        x = tf.nn.depthwise_conv2d(x, w, [1,stride,stride,1], padding)
        
        if norm=='batch_norm':
            x = tf.layers.batch_normalization(x, training=is_training, epsilon=0.001,name='batchnorm')
        elif norm=='group_norm':
            x = group_norm(x,name='groupnorm')
        else:
            b = tf.get_variable('bias',scale,tf.float32,initializer=tf.constant_initializer(0.001))
            x += b
        if activate=='leaky': 
            x = LeakyRelu(x,leak=0.1, name='leaky')
        elif activate=='selu':
            x = selu(x,name='selu')
        elif activate=='swish':
            x = swish(x,name='swish')
        elif activate=='relu':
            x = tf.nn.relu(x,name='relu')
        elif activate=='relu6':
            x = tf.nn.relu6(x,name='relu6')
        elif activate=='prelu':
            x = prelu(x)
        else:
            pass
        return x
##_group_conv with channel shuffle use depthwise_conv2d
def _group_conv(name,x,group=4,num_filters=16,kernel_size=1,stride=1,padding='SAME',norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        C = x.shape.as_list()[-1]
        num_012 = tf.shape(x)[:3]
        assert C%group==0 and num_filters%group==0
        
        w = GetWeight('weight',[kernel_size,kernel_size,C,num_filters//group])
        x = tf.nn.depthwise_conv2d(x, w, [1,stride,stride,1], padding)
        
        x = tf.reshape(x,tf.concat([ [num_012[0]], tf.cast(num_012[1:3]/kernel_size,tf.int32), tf.cast([group, C//group, num_filters//group],tf.int32)],axis=-1))
        x = tf.reduce_sum(x,axis=4)
        x = tf.transpose(x,[0,1,2,4,3])
        x = tf.reshape(x,tf.concat([ [num_012[0]], tf.cast(num_012[1:3]/kernel_size,tf.int32), tf.cast([num_filters],tf.int32)],axis=-1))
        
        if norm=='batch_norm':
            x = tf.layers.batch_normalization(x, training=is_training, epsilon=0.001,name='batchnorm')
        elif norm=='group_norm':
            x = group_norm(x,name='groupnorm')
        else:
            pass
        if activate=='leaky': 
            x = LeakyRelu(x,leak=0.1, name='leaky')
        elif activate=='selu':
            x = selu(x,name='selu')
        elif activate=='swish':
            x = swish(x,name='swish')
        elif activate=='relu':
            x = tf.nn.relu(x,name='relu')
        elif activate=='relu6':
            x = tf.nn.relu6(x,name='relu6')
        elif activate=='prelu':
            x = prelu(x)
        else:
            pass
        
        return x
##Attention
def Attention(name,x,norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        attention = SelfAttention('SelfAttention',x,norm,activate,is_training)
        weight = SE('SE',x,norm,activate,is_training)
        x = attention*(1+weight)
        return x
##selfattention
def SelfAttention(name,x,norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        C = x.get_shape().as_list()[-1]
        f = _B_conv_block('f',x,C//8,1,1,'SAME',norm,activate,is_training)
        g = _B_conv_block('g',x,C//8,1,1,'SAME',norm,activate,is_training)
        h = _B_conv_block('h',x,C,1,1,'SAME',norm,activate,is_training)
        
        f = tf.transpose(f,[0,3,2,1])
        g = tf.transpose(g,[0,3,1,2])
        h = tf.transpose(h,[0,3,1,2])
        
        attention = tf.multiply(f,g)
        attention = tf.reduce_mean(attention,[1],keep_dims=True)
        attention = tf.nn.softmax(attention)
        
        weight = tf.Variable(0.0,trainable=True)
        attention = weight*tf.multiply(h,attention)
        attention = tf.transpose(attention,[0,2,3,1])
        
        return x+attention
##senet
def SE(name,x,norm='group_norm',activate='selu',is_training=True):
    with tf.variable_scope(name):
        #none,none,none,C
        C = x.get_shape().as_list()[-1]
        #SEnet channel attention
        weight_c = tf.reduce_mean(x,[1,2],keep_dims=True)#none,1,1,C
        weight_c = _B_conv_block('conv_1',weight_c,C//8,1,1,'SAME',norm,activate,is_training)
        weight_c = _B_conv_block('conv_2',weight_c,C,1,1,'SAME',norm,activate,is_training)
        
        weight_c = tf.nn.sigmoid(weight_c)#none,1,1,C
        
        return weight_c
##weight variable
def GetWeight(name,shape,weights_decay = 0.00004):
    with tf.variable_scope(name):
        #w = tf.get_variable('weight',shape,tf.float32,initializer=VarianceScaling())
        w = tf.get_variable('weight',shape,tf.float32,initializer=glorot_uniform_initializer())
        weight_decay = tf.multiply(tf.nn.l2_loss(w), weights_decay, name='weight_loss')
        tf.add_to_collection('regularzation_loss', weight_decay)
        return w
##initializer
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops
import math
def glorot_uniform_initializer(seed=None, dtype=dtypes.float32):
    return VarianceScaling(scale=1.0,
                          mode="fan_avg",
                          distribution="uniform",
                          seed=seed,
                          dtype=dtype)
def glorot_normal_initializer(seed=None, dtype=dtypes.float32):
    return VarianceScaling(scale=1.0,
                          mode="fan_avg",
                          distribution="normal",
                          seed=seed,
                          dtype=dtype)
def _compute_fans(shape):
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        receptive_field_size = 1.
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out
class VarianceScaling():
    def __init__(self, scale=1.0,
                 mode="fan_in",
                 distribution="normal",
                 seed=None,
                 dtype=dtypes.float32):
      if scale <= 0.:
          raise ValueError("`scale` must be positive float.")
      if mode not in {"fan_in", "fan_out", "fan_avg"}:
          raise ValueError("Invalid `mode` argument:", mode)
      distribution = distribution.lower()
      if distribution not in {"normal", "uniform"}:
          raise ValueError("Invalid `distribution` argument:", distribution)
      self.scale = scale
      self.mode = mode
      self.distribution = distribution
      self.seed = seed
      self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
      if dtype is None:
          dtype = self.dtype
      scale = self.scale
      scale_shape = shape
      if partition_info is not None:
          scale_shape = partition_info.full_shape
      fan_in, fan_out = _compute_fans(scale_shape)
      if self.mode == "fan_in":
          scale /= max(1., fan_in)
      elif self.mode == "fan_out":
          scale /= max(1., fan_out)
      else:
          scale /= max(1., (fan_in + fan_out) / 2.)
      if self.distribution == "normal":
          stddev = math.sqrt(scale)
          return random_ops.truncated_normal(shape, 0.0, stddev,
                                             dtype, seed=self.seed)
      else:
          limit = math.sqrt(3.0 * scale)
          return random_ops.random_uniform(shape, -limit, limit,
                                           dtype, seed=self.seed)
##group_norm
def _max_divisible(input,max=1):
    for i in range(1,max+1)[::-1]:
        if input%i==0:
            return i
def group_norm(x, eps=1e-5, name='group_norm') :
    with tf.variable_scope(name):
        _, _, _, C = x.get_shape().as_list()
        G = _max_divisible(C,max=C//2+1)
        G = min(G, C)
        if C%32==0:
            G = min(G,32)
        
        #group_list = tf.split(tf.expand_dims(x,axis=3),num_or_size_splits=G,axis=4)#[(none,none,none,1,C//G),...]
        #x = tf.concat(group_list,axis=3)#none,none,none,G,C//G
        x = tf.reshape(x,tf.concat([tf.shape(x)[:3],tf.constant([G,C//G])],axis=0))#none,none,none,G,C//G
        
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)#none,none,none,G,C//G
        x = (x - mean) / tf.sqrt(var + eps)#none,none,none,G,C//G
        
        #group_list = tf.split(x,num_or_size_splits=G,axis=3)#[(none,none,none,1,C//G),...]
        #x = tf.squeeze(tf.concat(group_list,axis=4),axis=3)#none,none,none,C
        x = tf.reshape(x,tf.concat([tf.shape(x)[:3],tf.constant([C])],axis=0))#none,none,none,C

        gamma = tf.Variable(tf.ones([C]), name='gamma')
        beta = tf.Variable(tf.zeros([C]), name='beta')
        gamma = tf.reshape(gamma, [1, 1, 1, C])
        beta = tf.reshape(beta, [1, 1, 1, C])

    return x* gamma + beta
##LeakyRelu
def LeakyRelu(x, leak=0.1, name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)
##selu
def selu(x,name='selu'):
    with tf.variable_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)
##swish
def swish(x,name='swish'):
    with tf.variable_scope(name):
        beta = tf.Variable(1.0,trainable=True)
        return x*tf.nn.sigmoid(beta*x)
##crelu 注意使用时深度要减半
def crelu(x,name='crelu'):
    with tf.variable_scope(name):
        x = tf.concat([x,-x],axis=-1)
        return tf.nn.relu(x)
def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs-abs(inputs))*0.5
    return pos + neg
################################################################################################################
################################################################################################################
################################################################################################################

if __name__=='__main__':
    import time
    from functools import reduce
    from operator import mul

    def get_num_params():
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params
    
    dataset_root = 'F:\\Learning\\tensorflow\\detect\\Dataset\\'
    
    anchors = [18,27, 28,75, 49,132, 55,43, 65,227, 84,86, 108,162, 109,288, 162,329, 174,103, 190,212, 245,348, 321,150, 343,256, 372,379]
    max_box_per_image = 60
    min_input_size = 224
    max_input_size = 224
    batch_size = 1
    
    train_ints, valid_ints, labels = create_training_instances(
        dataset_root+'VOC2012\\Annotations\\',
        dataset_root+'VOC2012\\JPEGImages\\',
        'data.pkl',
        '','','',
        ['person','head','hand','foot','aeroplane','tvmonitor','train','boat','dog','chair',
         'bird','bicycle','bottle','sheep','diningtable','horse','motorbike','sofa','cow',
         'car','cat','bus','pottedplant']
    )
    def normalize(image):
        return image/255.
    train_generator = BatchGenerator(
        instances           = train_ints, 
        anchors             = anchors,   
        labels              = labels,        
        downsample          = 32, 
        max_box_per_image   = max_box_per_image,
        batch_size          = batch_size,
        min_net_size        = min_input_size,
        max_net_size        = max_input_size,   
        shuffle             = True, 
        jitter              = 0.3, 
        norm                = normalize
    )
    valid_generator = BatchGenerator(
        instances           = valid_ints, 
        anchors             = anchors,   
        labels              = labels,        
        downsample          = 32, 
        max_box_per_image   = max_box_per_image,
        batch_size          = batch_size,
        min_net_size        = min_input_size,
        max_net_size        = max_input_size,   
        shuffle             = True, 
        jitter              = 0.0, 
        norm                = normalize
    )
    
    model = DynamicsMobilenetv2(num_classes=len(labels),
                      num_anchors=5,
                      batch_size = batch_size,
                      max_box_per_image = max_box_per_image,
                      max_grid=[max_input_size,max_input_size],
                      )
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        num_params = get_num_params()
        print('all params:{}'.format(num_params))
        
        batch = 0
        for input_list,dummy_yolo in train_generator.next():
            x_batch, anchors_batch,t_batch, yolo_1, yolo_2, yolo_3 = input_list
            dummy_yolo_1, dummy_yolo_2, dummy_yolo_3 = dummy_yolo
            '''
            feed_dict={model.input_image:x_batch,
                       model.anchors:anchors_batch,
                       model.is_training:1.0,
                       model.true_boxes:t_batch,
                       model.true_yolo_1:yolo_1,
                       model.true_yolo_2:yolo_2,
                       model.true_yolo_3:yolo_3}
            '''
            feed_dict={model.input_image:x_batch,
                       model.is_training:1.0}
            
            start = time.time()
            out = sess.run(model.classifier_logits,feed_dict=feed_dict)
            print('Spend Time:{}'.format(time.time()-start))
            
            print(out.shape)
            
            if batch==0:
                break