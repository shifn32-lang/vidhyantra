"""Thin wrapper around the GitHub REST API for the /AI/ GitHub-mode feature.
Every call takes the caller's own personal access token explicitly — nothing
here reads settings or caches a token, keeping this module a pure client."""
import base64

import requests

GITHUB_API = 'https://api.github.com'
_TIMEOUT = 20

# Paths a commit is never allowed to touch or even read, regardless of what
# the model proposes — CI workflows are a classic supply-chain injection
# point; settings/migrations/manage.py/Procfile/requirements.txt are
# app-critical infrastructure a wrong or adversarially-prompted AI edit
# could break or use to escalate (e.g. settings.py holds live API keys);
# and secrets/the local dev database have no business being read or
# rewritten through this path. Checked by basename as well as full path, so
# a nested copy (app/settings.py, migrations/0001_initial.py) is still
# caught, not just a top-level one.
_BLOCKED_PREFIXES = ('.github/workflows/',)
_BLOCKED_EXACT = {'db.sqlite3', 'manage.py', 'procfile', 'requirements.txt'}
_BLOCKED_SUFFIXES = ('settings.py', '.pem', '.key', '.pfx', '.p12')
_BLOCKED_SEGMENTS = {'migrations'}


class GitHubAPIError(Exception):
    pass


def is_path_blocked(path):
    norm = (path or '').strip().replace('\\', '/').lstrip('/')
    if not norm or '..' in norm.split('/'):
        return True
    segments = norm.split('/')
    basename = segments[-1].lower()
    if norm in _BLOCKED_EXACT or basename in _BLOCKED_EXACT:
        return True
    if norm == '.env' or norm.startswith('.env.') or '/.env' in norm:
        return True
    if any(seg.lower() in _BLOCKED_SEGMENTS for seg in segments[:-1]):
        return True
    if basename.endswith(_BLOCKED_SUFFIXES):
        return True
    return any(norm.startswith(p) for p in _BLOCKED_PREFIXES)


def _headers(token):
    return {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
    }


def _request(method, token, path, **kwargs):
    try:
        resp = requests.request(method, f'{GITHUB_API}{path}', headers=_headers(token), timeout=_TIMEOUT, **kwargs)
    except requests.RequestException as e:
        raise GitHubAPIError(f'Could not reach GitHub: {e}')
    if resp.status_code >= 400:
        detail = resp.text
        try:
            detail = resp.json().get('message', detail)
        except ValueError:
            pass
        raise GitHubAPIError(f'{resp.status_code} {detail}')
    return resp


def get_authenticated_user(token):
    return _request('GET', token, '/user').json()


def list_user_repos(token):
    repos = []
    for page in range(1, 6):  # cap at 500 repos — plenty for a personal account
        batch = _request('GET', token, '/user/repos', params={'per_page': 100, 'page': page, 'sort': 'updated'}).json()
        repos.extend(batch)
        if len(batch) < 100:
            break
    return [
        {
            'full_name': r['full_name'], 'private': r['private'], 'default_branch': r['default_branch'],
            # Shown in the repository picker so similarly named repos are easy to tell apart.
            'description': (r.get('description') or '')[:140],
            'language': r.get('language') or '',
            'updated_at': r.get('updated_at') or '',
            'archived': bool(r.get('archived')),
            'fork': bool(r.get('fork')),
        }
        for r in repos
    ]


def get_repo(token, owner, repo):
    return _request('GET', token, f'/repos/{owner}/{repo}').json()


def get_tree(token, owner, repo, branch):
    ref = _request('GET', token, f'/repos/{owner}/{repo}/git/refs/heads/{branch}').json()
    sha = ref['object']['sha']
    tree = _request('GET', token, f'/repos/{owner}/{repo}/git/trees/{sha}', params={'recursive': '1'}).json()
    return [t['path'] for t in tree.get('tree', []) if t.get('type') == 'blob']


def get_file(token, owner, repo, path, branch):
    data = _request('GET', token, f'/repos/{owner}/{repo}/contents/{path}', params={'ref': branch}).json()
    content = base64.b64decode(data['content']).decode('utf-8', errors='replace')
    return content, data['sha']


def upsert_file(token, owner, repo, path, content, message, branch, sha=None):
    body = {
        'message': message,
        'content': base64.b64encode(content.encode('utf-8')).decode('ascii'),
        'branch': branch,
    }
    if sha:
        body['sha'] = sha
    return _request('PUT', token, f'/repos/{owner}/{repo}/contents/{path}', json=body).json()


def delete_file(token, owner, repo, path, message, branch, sha):
    body = {'message': message, 'sha': sha, 'branch': branch}
    return _request('DELETE', token, f'/repos/{owner}/{repo}/contents/{path}', json=body).json()


def get_branch_sha(token, owner, repo, branch):
    ref = _request('GET', token, f'/repos/{owner}/{repo}/git/refs/heads/{branch}').json()
    return ref['object']['sha']


def create_branch(token, owner, repo, new_branch, from_sha):
    body = {'ref': f'refs/heads/{new_branch}', 'sha': from_sha}
    return _request('POST', token, f'/repos/{owner}/{repo}/git/refs', json=body).json()


def delete_branch(token, owner, repo, branch):
    # Best-effort cleanup of a working branch that ended up with nothing
    # applied to it — never raises, since a leftover empty branch is
    # harmless and not worth failing the whole request over.
    try:
        _request('DELETE', token, f'/repos/{owner}/{repo}/git/refs/heads/{branch}')
    except GitHubAPIError:
        pass


def create_pull_request(token, owner, repo, title, head, base, body=''):
    payload = {'title': title[:250] or 'AI-proposed change', 'head': head, 'base': base, 'body': body}
    return _request('POST', token, f'/repos/{owner}/{repo}/pulls', json=payload).json()
