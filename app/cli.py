import argparse
import getpass
import re
from .db import init_db,connect
from .security import hash_password

def main():
    parser=argparse.ArgumentParser(description='SIMS-AI administration')
    parser.add_argument('command',choices=['create-admin','reset-password'])
    parser.add_argument('username')
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]{3,64}',args.username):
        parser.error('Username: 3-64 ASCII letters, digits, _, ., -')
    password=getpass.getpass('Password (12+ characters): ')
    if password != getpass.getpass('Repeat password: '):
        parser.error('Passwords differ')
    encoded=hash_password(password)
    init_db()
    with connect() as db:
        if args.command=='create-admin':
            db.execute("INSERT INTO users(username,password,role) VALUES (?,?,'admin')",(args.username,encoded))
        else:
            cur=db.execute('UPDATE users SET password=? WHERE username=?',(encoded,args.username))
            if not cur.rowcount:
                parser.error('User not found')
            db.execute('DELETE FROM sessions WHERE user_id=(SELECT id FROM users WHERE username=?)',(args.username,))
    print('Done')

if __name__=='__main__':
    main()
