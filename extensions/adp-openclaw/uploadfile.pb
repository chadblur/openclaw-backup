rpc DescribeRemoteBotStorageCredential(DescribeRemoteBotStorageCredentialReq) returns (DescribeRemoteBotStorageCredentialRsp) {}; // 查询文件存储密钥

message DescribeRemoteBotStorageCredentialReq {
  string token = 1 [(validate.rules).string = { min_len: 1 }]; // 机器人token
  string file_type = 2; // 文件类型
}

message DescribeRemoteBotStorageCredentialRsp {
  message Credentials {
    // token
    string token = 1;
    // 临时证书密钥 ID
    string tmp_secret_id = 2;
    // 临时证书密钥 Key
    string tmp_secret_key = 3;
  }
  // 密钥信息
  Credentials credentials = 1;
  // 失效时间
  uint32 expired_time = 2;
  // 起始时间
  uint32 start_time = 3;
  // 对象存储 桶
  string bucket = 4;
  // 对象存储 可用区
  string region = 5;
  // 文件目录
  string file_path = 6;
  // 上传url
  string upload_url = 7;
  // 下载url
  string file_url = 8;
}