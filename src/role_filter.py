# Role-based filtering system for the AI Query System 

from typing import List, Dict, Any, Optional

class RoleFilter:
    """Simple role-based filtering for responses"""
    
    def __init__(self):
        self.role_permissions = {
            'admin': {'all_documents': True, 'sensitive_info': True},
            'manager': {'all_documents': True, 'sensitive_info': False},
            'employee': {'all_documents': False, 'sensitive_info': False},
            'public': {'all_documents': False, 'sensitive_info': False}
        }
        
        self.document_restrictions = {
            'cybersecurity.txt': ['admin', 'manager'],  # Restricted to admin and manager
            'cloud_devops.txt': ['admin', 'manager', 'employee'],  # Not for public
        }
        
        self.sensitive_keywords = [
            'security', 'password', 'encryption', 'vulnerability', 
            'attack', 'penetration', 'firewall', 'intrusion'
        ]
    
    def filter_documents(self, documents: List[Dict[str, Any]], user_role: str = 'public') -> List[Dict[str, Any]]:
        """Filter documents based on user role"""
        if user_role not in self.role_permissions:
            user_role = 'public'
        
        if self.role_permissions[user_role]['all_documents']:
            return documents
        
        filtered_docs = []
        for doc in documents:
            source = doc.get('source', '')
            allowed_roles = self.document_restrictions.get(source, ['admin', 'manager', 'employee', 'public'])
            
            if user_role in allowed_roles:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def filter_response(self, response: str, user_role: str = 'public') -> str:
        """Filter sensitive information from response based on user role"""
        if user_role not in self.role_permissions:
            user_role = 'public'
        
        if self.role_permissions[user_role]['sensitive_info']:
            return response
        
        # For non-privileged users, replace sensitive information
        filtered_response = response
        for keyword in self.sensitive_keywords:
            if keyword.lower() in filtered_response.lower():
                filtered_response = filtered_response.replace(
                    keyword, 
                    f"[{keyword.upper()}_INFO_RESTRICTED]"
                )
        
        # Add disclaimer for filtered content
        if filtered_response != response:
            filtered_response += "\n\n[Note: Some sensitive information has been filtered based on your access level.]"
        
        return filtered_response


class FeedbackAnalyzer:
    """Analyze feedback to improve system performance"""
    
    def __init__(self, feedback_file: str = "feedback.jsonl"):
        self.feedback_file = feedback_file
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics from collected feedback"""
        import json
        import os
        
        if not os.path.exists(self.feedback_file):
            return {
                'total_feedback': 0,
                'helpful_percentage': 0,
                'common_issues': [],
                'suggestions': []
            }
        
        feedback_data = []
        try:
            with open(self.feedback_file, 'r') as f:
                for line in f:
                    if line.strip():
                        feedback_data.append(json.loads(line))
        except Exception as e:
            print(f"Error reading feedback: {e}")
            return {'error': str(e)}
        
        if not feedback_data:
            return {
                'total_feedback': 0,
                'helpful_percentage': 0,
                'common_issues': [],
                'suggestions': []
            }
        
        total = len(feedback_data)
        helpful_count = sum(1 for f in feedback_data if f.get('helpful', False))
        helpful_percentage = (helpful_count / total) * 100 if total > 0 else 0
        
        # Extract common issues from comments
        comments = [f.get('comments', '') for f in feedback_data if f.get('comments')]
        common_words = self._extract_common_words(comments)
        
        return {
            'total_feedback': total,
            'helpful_percentage': round(helpful_percentage, 1),
            'helpful_count': helpful_count,
            'unhelpful_count': total - helpful_count,
            'common_issues': common_words[:5],  # Top 5 common words in feedback
            'recent_feedback': feedback_data[-5:] if len(feedback_data) >= 5 else feedback_data
        }
    
    def _extract_common_words(self, comments: List[str]) -> List[str]:
        """Extract common words from feedback comments"""
        if not comments:
            return []
        
        word_count = {}
        for comment in comments:
            words = comment.lower().split()
            for word in words:
                # Filter out common words
                if len(word) > 3 and word not in ['this', 'that', 'with', 'from', 'they', 'have', 'were', 'been', 'their']:
                    word_count[word] = word_count.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words]


if __name__ == "__main__":
    # Test role filtering
    role_filter = RoleFilter()
    
    test_docs = [
        {'source': 'ai_ml_basics.txt', 'text': 'AI content'},
        {'source': 'cybersecurity.txt', 'text': 'Security content'},
        {'source': 'data_science.txt', 'text': 'Data science content'}
    ]
    
    print("Testing role-based filtering:")
    for role in ['public', 'employee', 'manager', 'admin']:
        filtered = role_filter.filter_documents(test_docs, role)
        print(f"{role}: {len(filtered)} documents allowed")
    
    # Test response filtering
    test_response = "This involves password security and firewall configuration."
    filtered_response = role_filter.filter_response(test_response, 'public')
    print(f"\nOriginal: {test_response}")
    print(f"Filtered: {filtered_response}")
    
    # Test feedback analyzer
    feedback_analyzer = FeedbackAnalyzer()
    stats = feedback_analyzer.get_feedback_stats()
    print(f"\nFeedback stats: {stats}")