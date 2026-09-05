<img width="1280" height="640" alt="git (1)" src="https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd" />

# Roast Report 🎯

## Basic Details Your group chat already knows who's the problem. We just made it say it out loud.
### Team Name: [Your Team Name]

### Team Members
- Team Lead: [Name] - [College]
- Member 2: [Name] - [College]
- Member 3: [Name] - [College]

### Project Description
Roast Report puts your group chat on trial. Drop in a real WhatsApp or Telegram export and it hands back a "case file" of who leaves you on seen, who double-texts, who yaps, and who spams the same emoji into oblivion — with actual quoted messages as evidence.

### The Problem (that doesn't exist)
Somewhere in every group chat, there is a truth everyone privately suspects but nobody has the receipts to prove: someone is the problem. We all argue "you always leave me on seen!" and "you're always doubletexting, me " with zero empirical basis — pure vibes, no evidence, no defendant ever properly convicted. This is, objectively, an urgent crisis of unaccountability, and absolutely nobody asked us to fix it...

### The Solution (that nobody asked for)
We built a courtroom out of your own chat export. It parses your real messages, scores everyone's reply gaps, double-texts, message length, and emoji habits against the group's own baseline (real z-scores, not vibes), crowns the worst offender in each category, and stacks "combo" charges for people who commit multiple crimes at once. Every accusation comes with a quoted message as Exhibit A. It started as a fake ML model that predicted a red flag label we invented ourselves — we scrapped that and pointed it at real chats instead, because that was funnier.

## Technical Details
**Python version** Uses python to provide a more statistical result 
**HTML version** Clean and easy to read and analyze version 
### Technologies/Components Used
For Software:
- **Languages:** Python, JavaScript, HTML/CSS
- **Frameworks:** None — the browser interface is dependency-free vanilla JS, 
- **Libraries:** pandas, NumPy, scikit-learn (Logistic Regression, Decision Tree, Isolation Forest), Matplotlib, Seaborn
- **Tools:** VS Code, Google Fonts (Baloo 2, Gochi Hand)

For Hardware:
N/A — software only.

### Implementation
*There are two ways to run this project* — a Python CLI version (parsing + full ML pipeline), 
 a browser version (same logic, reimplemented in JS, zero install). They're independent of each other; you only need one.

# Installation
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
(No installation needed for the browser interface — it's a single self-contained `.html` file.)

# Run
```bash
python project2.py
```
Edit the config block at the top of the file to point at your own WhatsApp `.txt` or Telegram `.json` export (or use one of the included sample chats), then run.

For the browser interface, just open `case_file_handbook.html` in any browser — no server needed.

### Project Documentation
For Software:

# Screenshots (Add at least 3)
![Screenshot1](Add screenshot of the "start here" upload screen)
*The drag-and-drop file intake and sample-chat buttons — this is the entry point before any chat is loaded.*

![Screenshot2](Add screenshot of the rap sheet section)
*The per-person leaderboard: charges filed against each sender, their punishment score, and the "Most Wanted" badge on the worst offender.*

![Screenshot3](Add screenshot of the exhibits section)
*Real quoted messages used as evidence for the emoji-spam charges — Exhibit A, straight from the chat.*

# Diagrams
![Workflow](Add a flowchart: chat export → parser → feature extraction → z-score ranking → roast report)
*How a raw chat export becomes a scored, roasted leaderboard: parse messages → extract behavioral features → rank senders by z-score against the group baseline → render charges, combos, and punishment scores.*

For Hardware:
N/A — software only.

### Project Demo
# Video
[Add your demo video link here]
*Should show loading a sample chat and walking through the rap sheet, sentencing scale, and exhibits live.*

# Additional Demos
[Add a link to a live-hosted version of case_file_handbook.html, or attach the sample chat files used in the demo, if useful]

## Team Contributions
- Abhinand cs: programmer
- Abhinand cs: HTML design

---
Made with ❤️ at TinkerHub Useless Projects

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--26-26?link=https%3A%2F%2Ftinkerhub.org%2Fevents%2F1M8ORET9A1%2Fuseless-projects-3.0)
