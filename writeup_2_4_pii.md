# 2.4 Personally Identifiable Information

## (a)-(c) 实现说明

我在 `cs336_data/pii.py` 中实现了三个函数：

- `mask_emails(text)`：用正则匹配常见邮箱地址，并替换为 `|||EMAIL_ADDRESS|||`。
- `mask_phone_numbers(text)`：匹配常见美国电话号码格式，例如连续 10 位数字、`(283)-182-3829`、`(283) 182 3829` 和 `283-182-3829`，并替换为 `|||PHONE_NUMBER|||`。
- `mask_ips(text)`：匹配合法 IPv4 地址，每个 octet 限制在 `0..255`，并替换为 `|||IP_ADDRESS|||`。

三个函数都返回 `(masked_text, num_masked)`。我也在 `tests/adapters.py` 中接好了对应的 `run_mask_emails`、`run_mask_phone_numbers` 和 `run_mask_ips` adapter。

验证命令：

```sh
.venv/bin/python -m pytest tests/test_pii.py
```

结果：`5 passed`。

## (d) 朴素 PII mask 的下游问题

如果把这些过滤器朴素地应用到整个训练集，语言模型可能会学到大量特殊占位符模式，从而在生成时过度输出 `|||EMAIL_ADDRESS|||`、`|||PHONE_NUMBER|||` 这类字符串，而不是自然文本。正则还会产生 false positives，例如把产品型号、备案号、长数字域名或技术配置误当成电话号码，从而破坏原本有用的内容；也会产生 false negatives，例如漏掉非美国电话号码、写法很怪的邮箱、IPv6 地址或被 HTML/空格拆开的 PII。缓解方式包括：按文档类型和语言做分层评估，使用更成熟的 PII 检测库或多阶段检测器，在训练时控制占位符频率，对敏感高风险数据直接删除整段而不是只替换，并保留人工审计样本来估计误报和漏报。

## (e) WARC 样本上的 PII masking 观察

我运行了 `scripts/analyze_pii_masking.py`，对 `local-shared-data/CC/example.warc.gz` 的前 500 个 response record 先使用 2.2 的 HTML 抽取函数得到文本，再依次应用 email、phone 和 IPv4 mask。前 500 个抽取文本中有 129 个文档至少发生了一次替换；我用固定随机种子从这些文档中抽取了 20 个样本人工查看。

大部分 email 替换看起来是正确的，常见于网站页脚、联系页面、博客评论区和采购公告，例如俄文地图站、丹麦体育馆页面、中文招标公司页面、越南地产页面里的公开联系邮箱都被替换。电话号码也能覆盖一些常见格式，例如中文招聘站和美国康复机构页面的联系号码被成功替换。

主要 false positives 来自“看起来像电话号码的非电话数字”。例如，一个中文企业站的数字域名被部分替换，留下了 `.cn` 后缀；另一个中文公司页面的ICP备案号被替换成了 `|||PHONE_NUMBER|||`；一些公告或产品页面中的编号也有被误判的风险。主要 false negatives 来自美国格式以外的号码：例如俄罗斯 `+7`、斯洛伐克 `+421`、捷克 `+420`、中国手机或带不同分组方式的座机号码在样本中经常没有被遮掉。IPv4 在这 20 个随机替换样本中没有出现，因此这个样本不能很好评估 IP mask 的误报和漏报；单元测试覆盖了合法 IPv4 的基本行为。
