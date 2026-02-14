---
layout: default
---

# 欢迎来到 inaniloquent 的主页 👋

这里记录了我在技术探索过程中的一些思考、踩坑记录与代码总结。

## 📝 最近的文章

<ul>
  {% for post in site.posts %}
    <li style="margin-bottom: 8px;">
      <a href="{{ site.baseurl }}{{ post.url }}" style="font-weight: 600;">{{ post.title }}</a>
      <span style="color: #57606a; font-size: 0.9em; margin-left: 8px;">{{ post.date | date: "%Y-%m-%d" }}</span>
    </li>
  {% endfor %}
</ul>

## 👨‍💻 关于我

目前主要关注 **AI Infra** 开发、**C++** 底层架构以及基于 **Go** 的后端系统。同时我也活跃在开源社区，参与飞桨（PaddlePaddle）等开源项目的建设。