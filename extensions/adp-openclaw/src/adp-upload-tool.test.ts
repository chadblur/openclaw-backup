/**
 * ADP Upload Tool 测试文件
 * 
 * 使用方式:
 * 
 * 1. 使用环境变量（推荐，与插件配置一致）:
 *    ADP_OPENCLAW_CLIENT_TOKEN=your-token npx tsx src/adp-upload-tool.test.ts <filePath>
 * 
 * 2. 使用命令行参数:
 *    npx tsx src/adp-upload-tool.test.ts --token <botToken> <filePath>
 */

import { AdpUploader, getStorageCredential } from "./adp-upload-tool.js";

async function main() {
  const args = process.argv.slice(2);
  
  let botToken: string | undefined;
  let filePaths: string[] = [];
  
  // 解析参数
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--token" && i + 1 < args.length) {
      botToken = args[i + 1];
      i++; // 跳过下一个参数
    } else if (!args[i].startsWith("--")) {
      filePaths.push(args[i]);
    }
  }
  
  // 如果没有通过参数传入 token，尝试从环境变量读取
  if (!botToken) {
    botToken = process.env.ADP_OPENCLAW_CLIENT_TOKEN;
  }

  if (filePaths.length === 0) {
    console.log("ADP Upload Tool 测试");
    console.log("");
    console.log("使用方式:");
    console.log("  1. 使用环境变量（推荐，与插件配置一致）:");
    console.log("     ADP_OPENCLAW_CLIENT_TOKEN=your-token npx tsx src/adp-upload-tool.test.ts <filePath>");
    console.log("");
    console.log("  2. 使用命令行参数:");
    console.log("     npx tsx src/adp-upload-tool.test.ts --token <botToken> <filePath>");
    console.log("");
    console.log("示例:");
    console.log("  npx tsx src/adp-upload-tool.test.ts --token my-bot-token ./test.txt");
    console.log("  ADP_OPENCLAW_CLIENT_TOKEN=my-token npx tsx src/adp-upload-tool.test.ts ./file1.txt ./file2.pdf");
    process.exit(1);
  }

  if (!botToken) {
    console.error("错误: 未提供 botToken");
    console.error("请通过 --token 参数或 ADP_OPENCLAW_CLIENT_TOKEN 环境变量设置");
    process.exit(1);
  }

  console.log("=".repeat(60));
  console.log("ADP Upload Tool 测试");
  console.log("=".repeat(60));
  console.log(`Bot Token: ${botToken.substring(0, 10)}...`);
  console.log(`文件列表: ${filePaths.join(", ")}`);
  console.log("");

  // 测试获取临时密钥
  console.log("1. 测试获取临时密钥...");
  try {
    const credential = await getStorageCredential(botToken);
    console.log("   ✓ 获取密钥成功");
    console.log(`   - Bucket: ${credential.bucket}`);
    console.log(`   - Region: ${credential.region}`);
    console.log(`   - File Path: ${credential.file_path}`);
    console.log(`   - Secret ID: ${credential.credentials.tmp_secret_id.substring(0, 10)}...`);
    console.log(`   - 有效期至: ${new Date(credential.expired_time * 1000).toLocaleString()}`);
    console.log("");
  } catch (error) {
    console.log("   ✗ 获取密钥失败");
    console.log(`   - 错误: ${error instanceof Error ? error.message : error}`);
    process.exit(1);
  }

  // 使用 AdpUploader 类测试上传
  console.log("2. 使用 AdpUploader 测试上传...");
  const uploader = new AdpUploader({ clientToken: botToken });
  console.log(`   - 配置状态: ${uploader.isConfigured() ? "已配置" : "未配置"}`);
  console.log(`   - Token 预览: ${uploader.getTokenPreview()}`);
  console.log("");

  if (filePaths.length === 1) {
    // 单文件上传
    console.log("3. 测试单文件上传...");
    const result = await uploader.upload(filePaths[0]);
    if (result.ok) {
      console.log("   ✓ 上传成功");
      console.log(`   - 下载链接: ${result.fileUrl}`);
    } else {
      console.log("   ✗ 上传失败");
      console.log(`   - 错误: ${result.error}`);
    }
  } else {
    // 批量上传
    console.log("3. 测试批量上传...");
    const results = await uploader.uploadMultiple(filePaths);
    results.forEach((result, index) => {
      const filePath = filePaths[index];
      if (result.ok) {
        console.log(`   ✓ [${filePath}] 上传成功`);
        console.log(`     下载链接: ${result.fileUrl}`);
      } else {
        console.log(`   ✗ [${filePath}] 上传失败`);
        console.log(`     错误: ${result.error}`);
      }
    });
  }

  console.log("");
  console.log("=".repeat(60));
  console.log("测试完成");
  console.log("=".repeat(60));
}

main().catch(console.error);
