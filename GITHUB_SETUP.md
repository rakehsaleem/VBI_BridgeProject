# GitHub Setup Instructions

## Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Repository name: `VBI_BridgeProject` (or your preferred name)
5. Description: "Vehicle-Bridge Interaction (VBI) analysis for bridge damage detection using CNN"
6. Choose **Public** or **Private** as needed
7. **DO NOT** initialize with README, .gitignore, or license (we already have these)
8. Click "Create repository"

## Step 2: Add Remote and Push

After creating the repository on GitHub, run these commands in your terminal:

```bash
# Add remote (replace USERNAME with your GitHub username)
git remote add origin https://github.com/USERNAME/VBI_BridgeProject.git

# Rename master branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 3: Add Collaborator/Organizer

### Method 1: Through GitHub Website (Recommended)
1. Go to your repository on GitHub: `https://github.com/USERNAME/VBI_BridgeProject`
2. Click on **Settings** tab
3. Click **Collaborators** (in the left sidebar under "Access")
4. Click **Add people** button
5. Enter the GitHub username or email of your collaborator
6. Select permission level:
   - **Read**: Can view the repository
   - **Write**: Can push changes (recommended for collaborator)
   - **Admin**: Can manage settings (for organizer role)
7. Click **Add [username] to this repository**
8. The collaborator will receive an email invitation

### Method 2: Using GitHub CLI (if installed)
```bash
gh repo add-collaborator USERNAME/VBI_BridgeProject COLLABORATOR_USERNAME --permission write
```

## Step 4: Verify Setup

```bash
# Check remote configuration
git remote -v

# Check collaborators (if you have the GitHub CLI)
gh repo view --json collaborators
```

## Important Notes

- The `.gitignore` file already excludes large `.mat` files and `Results/` folder
- Only code files are tracked, not the simulation data
- Collaborators can now clone, push, and contribute to the project

