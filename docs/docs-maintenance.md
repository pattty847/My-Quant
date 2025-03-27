# Documentation Maintenance Process

## Overview
This document outlines a systematic approach to keeping project documentation up-to-date, with a special focus on maintaining context for AI assistants.

## Session Documentation Workflow

### 1. Pre-Session Documentation
Before starting a coding session:
- Note what you plan to change
- Identify which docs might need updating
- Create a new entry in session_logs

### 2. During-Session Documentation
While coding:
- Add clear comments about significant changes
- Keep a running list of files modified
- Note any unexpected issues or discoveries

### 3. Post-Session Documentation
After completing a session:
- Fill out the session summary template
- Update the AI context file
- Commit both to the repository

### 4. Weekly Documentation Review
Set aside time weekly to:
- Review all session logs from the week
- Update formal documentation based on changes
- Remove outdated information
- Ensure consistency across docs

## Session Summary Template

```markdown
# Session Summary: [Date]

## Goals
- [What you set out to accomplish]

## Changes Made
- [List key changes with brief explanations]

## Files Modified
- [List files that were changed]

## New Features/Capabilities
- [List any new features or capabilities]

## Bugs Fixed
- [List any bugs that were fixed]

## Documentation Updates Needed
- [List documentation files that need updating]

## Next Steps
- [List planned next actions]
```

## AI Context File

The AI context file is a special document designed to give AI assistants a quick overview of the current project state. It should be updated after each significant session.

### Location
`/docs/ai_context.md`

### Content Sections
1. **Project Status**: Current state with last update date
2. **Recent Changes**: Most recent modifications
3. **System Components**: Major parts of the system
4. **Available Commands**: User interface capabilities
5. **Current Focus**: What you're working on now
6. **Implementation Notes**: Technical details
7. **Envisioned Improvements**: Planned enhancements
8. **File Organization**: How the codebase is structured

### Update Frequency
- After any significant change
- At minimum, weekly
- Before starting a new conversation with an AI assistant

## Recent Experience Notes

Our recent debugging session highlighted the importance of detailed documentation:

1. **Bug Documentation**: When fixing bugs, document both the symptoms and the root cause
2. **Edge Cases**: Note any edge cases discovered during debugging (e.g., pagination limits)
3. **Decision Records**: Document why certain approaches were chosen (e.g., direct calculation vs. caching)
4. **System Limitations**: Record any limitations discovered (e.g., exchange API restrictions)
5. **Interdependencies**: Note where components have tight coupling that caused issues

## Documentation Organization

Keep documentation organized in a clear directory structure:

```
/docs
  /architecture       # System design and components
  /user-guides        # How to use the system
  /developer-guides   # How to extend the system
  /api-reference      # Details of functions/methods
  /session-logs       # Your session summaries
  /future-plans       # Ideas and roadmap
  ai_context.md       # The AI assistant context file
```

## Best Practices

1. **Versioning**: Include "Last Updated" dates on all docs
2. **Consistency**: Use consistent terminology throughout
3. **Examples**: Include practical examples where possible
4. **Diagrams**: Update visual diagrams when architecture changes
5. **Cleanup**: Remove outdated information promptly
6. **Linkage**: Cross-reference related documentation
7. **Code-Doc Syncing**: Ensure documentation updates happen alongside code changes

## AI Assistance with Documentation

When working with AI assistants on documentation:

1. Share the AI context file at the start of conversations
2. Ask for help updating specific documents based on session summaries
3. Request documentation reviews to catch inconsistencies
4. Consider using AI to generate initial documentation drafts based on code
5. Have AI help create user guides and example scenarios
6. Use AI to help maintain consistency between code and documentation

By following this process, documentation will stay current and AI assistants will always have proper context for helping with the project.
