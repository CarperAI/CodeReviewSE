all: Badges.xml

# Badges.xml Comments.xml PostHistory.xml PostLinks.xml Posts.xml Tags.xml Users.xml Votes.xml

Badges.xml: codereview.stackexchange.com.7z
	[ ! -f Badges.xml ] && 7za x codereview.stackexchange.com.7z || true

codereview.stackexchange.com.7z: 
	wget -nc https://archive.org/download/stackexchange/codereview.stackexchange.com.7z

