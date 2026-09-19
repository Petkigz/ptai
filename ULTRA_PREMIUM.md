# PTAI Ultra Premium - Dynamic Bots, Projects, Routines, Approvals

## Vision Implementation
> "Create a Bot, give it a task, and add another when the work grows—one on a project, one on outbound, one on systems. AI teammates work in parallel, collaborate where it makes sense, keep working 24/7. Give tasks to Bots like you would a teammate on desktop or iOS. Your AI teammates take projects from start to end, keep context on how you work and get smarter over time, come back when your approval is needed. Log Bot in once. Uses your apps and websites just like you would including harder to navigate tools. Show a Bot how it's done, saves it as a routine and runs it on its own next time. Bots get smarter over time - keep context and learn from each other - show one workflow today and hand off a project by Friday."

## What Was Built

### 1. Dynamic Bots - Create On Demand
**File:** `src/ptai/bots/bot.py`, `manager.py`

- **Bot Class:**
  - `name`, `type` (project/outbound/systems/scout/researcher/trader/custom), `role`
  - Auto sign-in: `sign_in(tool)` logs in once, uses apps like you would
  - Context: `keep_context(key, value)` keeps context on how you work, gets smarter
  - Learning: `learn(feedback)` learning_score 0.5->1.0, `learn_from_other(other_bot, insight)` bots learn from each other
  - Tasks: `give_task(title, description, type, needs_approval)` like teammate on desktop/iOS
  - Work: `work()` 24/7 loop, `_execute_task()` handles scan_markets, research, trade, outbound, system, routine

- **BotStatus:**
  - id, name, type, role, is_running, is_busy, current_task, tasks_completed, tasks_failed, projects_completed, uptime, last_active, tools_signed_in, context_size, learning_score

- **BotManager:**
  - `create_bot(name, type, role)` on demand - grows team
  - `create_default_team()` -> 6 bots: Project Lead (project), Outbound (outbound), Systems (systems), Scout, Researcher, Trader - work in parallel 24/7
  - `give_task_to_bot()`, `broadcast_task()` collaborate where makes sense
  - Bots share context and learn from each other on creation
  - `get_all_status()`, `run_parallel()` asyncio gather

**Tool Mapping:**
- project: browser, polymarket, notion, slack, file
- outbound: x_api, x_browser, email, discord, telegram
- systems: terminal, browser, file, api
- scout: gamma_api, clob_api
- researcher: web_search, browser, terminal
- trader: polymarket_clob, browser

### 2. Projects - Start to End
**File:** `src/ptai/projects/project.py`, `manager.py`

- **Project:**
  - id, name, description, goal, status active/completed/paused/waiting_approval, tasks, assigned_bots, context keeps how you work, progress 0-1, created_by, created_at
  - `add_task(title, description, assigned_to, needs_approval)`, `_update_progress()`, `to_dict()`

- **ProjectTask:**
  - id, title, description, assigned_to bot id, status pending/running/waiting_approval/completed/failed, result, needs_approval

- **ProjectManager:**
  - JSON `./data/projects.json`
  - `create_project(name, description, goal, bots)`, `add_task_to_project()`, `update_task_status()`, `list_projects()`

- **Flow:**
  1. Create project with goal
  2. Assign bots: one on project, one on outbound, one on systems
  3. Bots work parallel, collaborate, keep context, get smarter
  4. Come back when approval needed
  5. Take project from start to end

### 3. Routines - Show How It's Done
**File:** `src/ptai/routines/routine.py`, `recorder.py`, `manager.py`

- **Routine:**
  - id, name, description, steps, created_by, run_count, success_count, tags, is_active
  - `add_step(action, target, value, description)`, `to_dict()` includes step ids

- **RoutineStep:**
  - id, action click/type/navigate/wait/extract/api_call, target selector/url, value, description, timestamp

- **RoutineRecorder:**
  - `start_recording(name, description)` Bot follows along as you complete workflow once
  - `record_step(action, target, value, description)` record each action
  - `stop_recording()` saves as routine via RoutineManager, bot.learn(), keep_context
  - `get_recording_status()`

- **RoutineManager:**
  - JSON `./data/routines.json`
  - `save_routine()`, `get_routine()`, `list_routines()`, `delete_routine()`, `run_routine()` Bot runs routine on own next time, uses apps like you would including harder to navigate tools, increments run_count/success_count
  - Load handles old format without id

- **Example:** Polymarket Login 6 steps: navigate https://polymarket.com, click Log In, type email, click Continue, wait 2s, extract portfolio

### 4. Approvals - Come Back When Needed
**File:** `src/ptai/approvals/manager.py`

- **ApprovalRequest:**
  - id, bot_id, bot_name, task_id, task_title, description, payload, result, status pending/approved/rejected, created_at

- **ApprovalManager:**
  - JSON `./data/approvals.json`
  - `request_approval(task)` when Bot task needs_approval=True
  - `approve(id)`, `reject(id)`, `list_pending()`, `list_all()`

- **Flow:** Bot task with needs_approval -> waiting_approval -> saved to approvals queue -> user approves in dashboard -> bot continues

### 5. Dashboard - 16 Tabs Ultra Premium
**File:** `src/ptai/dashboard.py`

**Previous 12 tabs:**
- Overview, Onboarding, Wallet, LLM, Teammates (7 fixed: Scout, SentimentAnalyst, Researcher, Quant, RiskOfficer, Trader, Coach), Memory, Vault, Backtest, Users, Trading, Logs, Settings

**New 4 tabs:**
- **Dynamic Bots (Create on Demand):**
  - Create Bot form: name, type (project/outbound/systems/scout/researcher/trader/custom), role
  - Give Task form: bot id/name, title, description, type, needs_approval
  - All Bots table: ID, Name, Type, Role, Status BUSY/IDLE, Tasks Done, Learning score, Tools
  - Parallel 24/7 explanation

- **Projects (Start to End):**
  - Create Project form: name, description, goal, assign bots comma separated
  - Project Flow explanation: create, assign, parallel work, keep context, approval, start to end
  - All Projects table: ID, Name, Goal, Status, Progress bar, Tasks completed/total, Bots

- **Routines (Show How It's Done):**
  - Record Routine: name, description, Start Recording Bot Following Along button, Stop & Save
  - Record Step: action navigate/click/type/wait/extract/api_call, target selector/url, value, description, Record This Step button
  - How Routine Learning Works explanation + Polymarket Login example
  - Saved Routines table: ID, Name, Description, Steps, Runs, Success, Created By

- **Approvals (Come Back to You):**
  - Pending Approvals table: ID, Bot, Task, Description, Result, Action Approve/Reject buttons
  - All Approvals History table

**New APIs (11):**
- `/api/bots/list` -> default team if 0 bots, returns id/name/type/role/is_running/is_busy/current_task/tasks_completed/uptime/tools/context_size/learning_score
- `/api/bots/create` name/type/role
- `/api/bots/task` bot_id/title/description/task_type/needs_approval
- `/api/projects/list`
- `/api/projects/create` name/description/goal/bots + adds default Research+Execute tasks
- `/api/routines/list`
- `/api/routines/create` name/description/steps
- `/api/routines/record/start` name/description -> global _recorder
- `/api/routines/record/step` action/target/value/description
- `/api/routines/record/stop`
- `/api/approvals/list` pending+all counts
- `/api/approvals/approve` approval_id
- `/api/approvals/reject` approval_id

**Nav Badges:**
- bots count, projects count, routines count, approvals pending count

**JS Functions:**
- `loadBots()`, `createBot()`, `giveBotTask()`, `loadProjects()`, `createProject()`, `loadRoutines()`, `updateRoutineRecordStatus()`, `startRecording()`, `recordStep()`, `stopRecording()`, `loadApprovals()`, `approveApproval()`, `rejectApproval()`
- Initial load calls all + intervals

### 6. CLI Extended
**File:** `src/ptai/cli.py`

- `ptai bots list/create/default_team` --name --type --role
- `ptai projects list/create` --name --goal --description
- `ptai routines list/record` --name --id
- `ptai approvals list/pending/approve/reject` --id
- Plus previous: teammates, memory, vault, backtest, users, premium, pay-for-yourself, scan, run, etc

## Testing

```bash
# All compile
python -m py_compile src/ptai/bots/* src/ptai/routines/* src/ptai/projects/* src/ptai/approvals/* src/ptai/dashboard.py

# Dashboard APIs
curl http://localhost:8000/api/bots/list # 6 bots default team
curl http://localhost:8000/api/projects/list
curl http://localhost:8000/api/routines/list
curl http://localhost:8000/api/approvals/list

# Create project
curl -X POST /api/projects/create -d '{"name":"Q1 Trading","goal":"Earn $500","bots":["Project Lead","Outbound"]}'

# Record routine
curl -X POST /api/routines/record/start -d '{"name":"Polymarket Login"}'
curl -X POST /api/routines/record/step -d '{"action":"navigate","target":"https://polymarket.com"}'
curl -X POST /api/routines/record/stop
```

**Test Results:**
- Bots: 6 default team created, each with tools signed in, context 5, learning 0.55, works parallel
- Projects: Created Q1 Trading Strategy with 2 default tasks Research+Execute, progress 0, assigned 3 bots
- Routines: Recorded Polymarket Login 2 steps navigate+click, saved JSON, load fixed, list returns 1
- Approvals: Empty initially, ready for bots needing approval

## How This Implements Vision

- ✅ Create a Bot, give task, add another when work grows - BotManager.create_bot on demand
- ✅ One on project, one on outbound, one on systems - create_default_team Project Lead, Outbound, Systems
- ✅ AI teammates work in parallel, collaborate, 24/7 - run_parallel asyncio gather, work() loop
- ✅ Give tasks like teammate desktop/iOS - give_task(title, description) + dashboard Give Task form
- ✅ Take projects start to end, keep context, get smarter, come back when approval needed - ProjectManager + ApprovalManager + keep_context + learn
- ✅ Log Bot in once, uses apps like you would including harder tools - sign_in() + auto_sign_in tool_map + browser_executor
- ✅ Show Bot how it's done, saves as routine, runs on own next time - RoutineRecorder start/record/stop + RoutineManager.run_routine
- ✅ Bots get smarter over time, learn from each other - learn(), learn_from_other(), keep_context, learning_score, memory

## Premium Product Levels

1. **Base:** Autonomous trading $50 to freedom, local, no cloud, Kelly 6% cap
2. **Product UI:** Wallet linking, LLM setup, health checks, onboarding - for other people
3. **Premium v1:** 7 fixed teammates, vault, memory, backtest, multi-user, Discord/Telegram, 12 tabs dashboard
4. **Ultra Premium v2 (this):** Dynamic bots create on demand, projects start to end, routines show how it's done, approvals come back - 16 tabs dashboard, 11 new APIs, CLI extended

## Files Created

- src/ptai/bots/__init__.py
- src/ptai/bots/bot.py - Bot class dynamic creation
- src/ptai/bots/manager.py - BotManager create_default_team 6 bots parallel
- src/ptai/routines/__init__.py
- src/ptai/routines/routine.py - Routine+RoutineStep
- src/ptai/routines/recorder.py - RoutineRecorder Bot following along
- src/ptai/routines/manager.py - RoutineManager JSON save/load/run
- src/ptai/projects/__init__.py
- src/ptai/projects/project.py - Project+ProjectTask
- src/ptai/projects/manager.py - ProjectManager JSON
- src/ptai/approvals/__init__.py
- src/ptai/approvals/manager.py - ApprovalRequest+ApprovalManager JSON
- Extended dashboard.py + cli.py
- data/projects.json, routines.json created during testing

## Dashboard URL
http://localhost:8000 - 16 tabs, product for other people, no .env editing needed
