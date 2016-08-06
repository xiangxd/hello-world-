import tensorflow as tf
worker1 = "10.43.175.140:2222"
worker2 = "10.43.175.140:2223"
worker_hosts = [worker1, worker2]
cluster_spec = tf.train.ClusterSpec({ "worker": worker_hosts})
server = tf.train.Server(cluster_spec, job_name="worker", task_index=0)
server.join()
