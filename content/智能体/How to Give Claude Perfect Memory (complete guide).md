---
title: "How to Give Claude Perfect Memory (complete guide)"
source: "https://x.com/aiedge_/status/2046966170868486512?s=46"
author:
  - "[[@aiedge_]]"
published: 2026-04-08
created: 2026-04-27
description: "I don't care what anyone tells you - by default, Claude's memory is basically useless. It frequently forgets context; you constantly have to..."
tags:
  - "clippings"
---
![Image](https://picbed-1254120015.cos.ap-shanghai.myqcloud.com/Image%206.jpg?imageSlim)

I don't care what anyone tells you - by default, Claude's memory is basically useless.

It frequently forgets context; you constantly have to re-explain yourself, and even after you do, it still often doesn't remember.

Sadly, most people have been living with these flaws for months, without knowing there is a better way.

I use Claude every single day. I literally have more Claude screen time than any other app on my Mac.

Because I use Claude so often, I can't afford to have my most-used work tool randomly forget context and important data.

In need of a Claude that is as sharp as possible, I went down the rabbit hole of looking for every possible memory solution.

Luckily, my research was successful.

I discovered three "layers" of memory systems, and they've made Claude significantly more powerful.

One that is easy and works well enough for 90%+ of Claude users.

Another one that takes ~60 minutes to set up, but changes how Claude operates entirely.

And the last one turns Claude into a self-evolving second brain, trained on all your data.

In today's article, I'm revealing all three of these systems and how I made my Claude way sharper.

Whether you've never touched Claude before or you're a power user, I'm confident that you'll find a system that fits your needs.

## Layer one: Basic Memory (Beginner)

This is where everyone should start - four quick wins that take minutes to set up and immediately improve every conversation you have with Claude.

**1\. Memory Editing Tool**

Go to Settings → Memory right now.

This is the most overlooked page in all of Claude, and most people have never opened it.

What you will find is everything Claude has stored about you (preferences, facts, habits, working styles, etc.) accumulated passively across every conversation you have ever had.

Left unmanaged, your memory quickly fills up with garbage.

**The fix is simple:** read through everything on this page. Delete anything outdated, inaccurate, or irrelevant. Then, manually add the context you actually want Claude to carry permanently.

Stick to the basics here (your role and basic preferences) - we'll dive into building highly specific systems soon.

![Image](https://picbed-1254120015.cos.ap-shanghai.myqcloud.com/Image%207.jpg?imageSlim)

Chat Memory

2\. **Project Instructions**

If you use Claude Projects (you should), you need to fill in your Project Instructions field.

![Image](https://picbed-1254120015.cos.ap-shanghai.myqcloud.com/Image%208.jpg?imageSlim)

Project Instructions

**My advice:** Create projects for all your most-used workflows, then voice-prompt all your context into a Google Doc and upload it as a PDF for each project.

3\. **Tell Claude Directly**

The simplest memory hack on this list. Mid-conversation, just tell Claude what you want it to remember.

Things like:

```text
"Remember that I never want [x]"
"Remember that my role is [x]"
"Update your memory with [x]"
"Remember that I prefer responses under 400 words."
```

Claude will store these in your memory immediately. You can also tell it to forget things: "Forget that I mentioned \[x\]."

4\. **Memory Imports & Exports**

If you have been using ChatGPT (or another LLM) and have built up significant context there, you do not have to start from scratch in Claude; you have two effective options to transfer context:

a) You can tell ChatGPT you are switching platforms and ask it to generate a memory export document: "I'm switching this project to Claude, give me a summary document..."

b) You can use Import/Export in Claude

In Settings → Memory, you can import full data from other LLMs

![Image](https://picbed-1254120015.cos.ap-shanghai.myqcloud.com/Image%209.jpg?imageSlim)

Export/Import

These four quick edits will suffice for 90%+ of you, and they make an immediate impact on how Claude responds.

However, the next section is for people who want a real system for taking Claude to the next level.

## Layer two: Context File System (Intermediate)

Layer 1 fixes the basic memory problems.

**Layer 2 builds something more powerful:** a file-based memory architecture that lives on your computer, loads automatically into Cowork and Claude Code.

**The concept is simple:** Instead of prompting Claude for context, you store all of that context in .MD desktop files that Claude has access to.

You can also attach these markdown files to any LLM or AI agent system.

Start a new desktop folder, label it **"Claude Master Folder",** and build these four markdown files within it (Claude can help you do this):

1. **Instructions.** **MD**

This file tells Claude all your rules & instructions:

Example: ## Who you are ## What you do ## Rules ## What good outputs look like **Important to include:** "Update Memory. MD with my preferences over time."

This line is crucial; it's how you get Claude to create a running memory log of your data in the second markdown file.

2\. **Memory. MD**

This is the "brain" of Claude, and it gets continuously updated over time.

Example: ## Preferences ## Corrections ## Patterns ## Decisions

Now, whenever you say something like "stop using em dashes," Claude will go into the memory file and update it.

3\. **Context. MD**

The specific context file for \[x\] project.

Obviously, what's in this markdown file will change depending on your specific project.

You can also just create a general "business context" or "life context" markdown mega file.

4\. **Archive Copies**

This one is purely protective but worth doing.

Claude will update your memory files automatically as you work. Occasionally, it overwrites something incorrectly or makes a change you did not intend.

Without a backup system, that context is gone.

The fix is simple. Once a week, copy your entire master folder with Instructions, Memory, Context, and everything else into a separate archive folder that Claude cannot access, and label it with the date.

If anything breaks or gets overwritten incorrectly, you can restore from the archive.

**This is what the final product should look like:**

<video preload="none" tabindex="-1" playsinline="" aria-label="Embedded video" poster="https://pbs.twimg.com/amplify_video_thumb/2044552580609806337/img/2gf3N_nE2N_me1LC.jpg" style="width: 100%; height: 100%; position: absolute; background-color: black; top: 0%; left: 0%; transform: rotate(0deg) scale(1.005);"><source type="video/mp4" src="blob:https://x.com/3f6f71bd-b39e-44eb-b4c4-eac24e458647"></video>

![](https://picbed-1254120015.cos.ap-shanghai.myqcloud.com/2gf3N_nE2N_me1LC%201.jpg?imageSlim)

4 Markdown Files: Final Product

As I mentioned above, you can just get Claude to create this for you.

Just create a new folder called "Claude Master Folder," attach it to a new Cowork chat, and paste this prompt:

```text
Go into my "Claude Master Folder" in my connected workspace and build these four markdown files inside it:

Instructions.md — includes sections for: Who You Are, What You Do, Rules, What Good Outputs Look Like, and a line telling Claude to update Memory.md with my preferences over time.
Memory.md — includes sections for: Preferences, Corrections, Patterns, Decisions, and Personal Context. Pre-fill with placeholder examples so I know what to add.
Context.md — includes sections for: About This Project/Business, Audience, Key People & Collaborators, Active Projects & Priorities, Tools & Stack, and Important Background/History. Use a template format with placeholders I can fill in.
Archive-Guide.md — a step-by-step guide explaining why to archive, how to do it weekly (duplicate the folder, rename with the date, move it somewhere Claude can't access), what to include, how to restore if something breaks, and where to store the backups.
```

So you have the system built - now, how do you actually use it?

Anytime you're working in Cowork/Claude Code, you can attach your Master Folder, and Claude will use this as a mini memory database.

It will edit the memory markdown file, leaving you with something you can attach to any LLM, new chat, or AI agent.

You can also manually update the .MD files and create new folders for specific projects within your Master Folder.

This system is a complete game-changer, but what I'm about to show you in **Layer three** takes this system even further.

## Layer three: AI Second Brain (Advanced)

This is the deepest level, and frankly, it is not for everyone.

It does require some initial setup and ongoing maintenance, but for those of you who build it, it truly is the best option for an advanced, detailed memory system with Claude.

There are two options depending on how you work.

The first option is easier, and I included it for those of you who want a "simple" AI second brain.

The second option is more advanced (but better overall), and requires 1-2 hours of dedicated building (don't worry, I'll guide you through it all).

Keep in mind that for your AI second brain memory vault to be effective, you actually have to spend time maintaining it and updating your databases.

**Option 1: Claude x Notion**

Connecting Claude to Notion is the highest-leverage thing you can do in 5 minutes.

Go to Claude → Settings → Connectors, then enable the Notion connector.

![Image](https://picbed-1254120015.cos.ap-shanghai.myqcloud.com/Image%201.png?imageSlim)

Notion Connector

Once connected, Claude can read your Notion workspace directly inside any chat.

Now all your tasks, CRMs, notes, tables, etc., are accessible and editable for Claude.

I recommend creating a new **"Memory Database"** where you store all your AI preferences, rules, and important AI context.

As you're working with Claude, you can say: "Send this to my Notion Memory Database."

You can then export this Notion data to other LLMs or AI platforms via a CSV file or by using the Notion MCP connector.

This setup is similar to what I covered in Layer Two, except you now have nice visuals with Notion's built-in board views, to-do lists, and more, and you unlock additional functionality.

I personally don't use this setup often (Option two below is just better), but I do occasionally use it to send and store valuable mega prompts I use in Claude:

![Image](https://picbed-1254120015.cos.ap-shanghai.myqcloud.com/Image%2010.jpg?imageSlim)

Claude x Notion: Final Product

**Option 2: Claude x Obsidian**

Obsidian is a tool that stores everything as plain Markdown files on your computer, making it a solid way to connect with Claude and build a second brain.

**The setup**

1. **Download Obsidian**

Go to obsidian.md and download the app.

Create a new Vault (think of this as a simple desktop folder where Claude Code will store and access your data).

**2\. Select Vault in Claude Cowork/Claude Code**

Open the Claude desktop app and click 'Select Folder.'

Point it at your Obsidian Vault folder. Claude now has direct read and write access to everything inside it.

**3\. Inject mega prompt**

Paste Andrej Karpathy's LLM Knowledge Base system prompt into the chatbox.

This is the instruction set that tells Claude Code how to build, maintain, and evolve your wiki over time.

The prompt is available here:

[gist.github.com/karpathy/442a6bf555914893e9891c11519de94f](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)

**4\. Feed it your data**

Drop in any existing notes, CSV files, article exports, or Notion exports to start populating your second brain.

Claude then ingests each source, extracts the key information, and integrates it into an evolving memory wiki.

**Final Product**

The final product is an AI second brain knowledge wiki that links ideas, notes, remembers ALL your data, and looks like this:

![Image](https://picbed-1254120015.cos.ap-shanghai.myqcloud.com/Image%2011.jpg?imageSlim)

Claude x Obsidian: Final Product

**Which one should you choose?**

Notion = fast, simple option

Obsidian = local storage, and you want Claude to have a deep understanding (this is the most advanced memory system I've personally found).

If you're going to build the Claude x Obsidian second brain, I recommend reading this first (more details):

> Apr 8

## Final Thoughts

So, there you have it, my three layers to building memory systems inside Claude that make it way sharper.

Again, layer one is the basic setup that yields immediate results (enough for most people).

Layer two requires some setup, but is extremely valuable.

And lastly, layer three will completely change how you use Claude.

I hope you found this article helpful. I post AI articles 2-3x/week, and all my content is hand-written, based on how I'm actually using AI (no AI slop).

If that's the style of content you like to see, follow me [@aiedge\_](https://x.com/@aiedge_) and more will be on your feed soon!

Lastly, if you can, please Like/Repost this article so others can see it💙