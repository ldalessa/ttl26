(package-initialize)
(add-to-list 'load-path "~/.emacs.d/elisp/")
(require 'package)
(require 'use-package)
(require 'cmake-mode)
(require 'lua-mode)
(require 'dockerfile-mode)

;;;------------------------------------------------------------------------;;;
;;; MELPA packages
;;;------------------------------------------------------------------------;;;
(add-to-list 'package-archives '("melpa" . "http://melpa.org/packages/") t)

;;;------------------------------------------------------------------------;;;
;;; Set modes and interface stuff					   ;;;
;;;------------------------------------------------------------------------;;;
(delete-selection-mode t)		; I like it... so sue me
(global-font-lock-mode t)		; enable syntax highlighting
(column-number-mode t)			; enable column numbers

(menu-bar-mode -1)			; no menus
(blink-cursor-mode nil)			; or blinking

(setq inhibit-startup-message t)	; don't show startup screen
(setq display-time-day-and-date nil)	; don't show the date and time
(setq make-backup-files nil)		; never actually used them
(setq auto-save-default nil)
(setq ring-bell-function		; instead of beeping
      (lambda () (message "pip...")))		

(fset 'yes-or-no-p 'y-or-n-p)		; yes/no questions become y/n

;;;------------------------------------------------------------------------;;;
;;; Set colors								   ;;;
;;;------------------------------------------------------------------------;;;
(load-theme 'wombat t)

;;;------------------------------------------------------------------------;;;
;;; Set defaults							   ;;;
;;;------------------------------------------------------------------------;;;
(setq-default major-mode 'text-mode)
(setq-default fill-column 80)
(setq-default comment-fill-column 80)
(setq-default indent-tabs-mode t)
(setq-default tab-width 4)
(setq-default comment-column 48)
(setq-default truncate-partial-width-windows t) ; word wrap
(setq-default scroll-step 1)			; smooth scrolling
(setq-default auto-fill-function 'do-auto-fill) ; turn on auto-fill-mode
(defvaralias 'c-basic-offset 'tab-width)

;;;------------------------------------------------------------------------;;;
;;; Set keybindings							   ;;;
;;;------------------------------------------------------------------------;;;
(global-set-key "\r"	      'newline-and-indent)
(global-set-key [?\M-g]	      'goto-line)
(global-set-key [?\C-c ?\C-c] 'comment-region)
(global-set-key [?\C-c ?\C-v] 'uncomment-region)
(global-set-key [?\M-p]	      'fill-paragraph)
(global-set-key [?\M-s]	      'fill-sentence)
(global-set-key [?\C-c \t]    'indent-for-comment) 
(global-set-key (kbd "C-x 9") (lambda () (interactive) ; sensible split 
				(split-window-right 86)))

(add-hook 'LaTeX-mode-hook
	  (lambda ()
	    (setq whitespace-style '(tabs tab-mark))
	    (flyspell-mode 1)
	    (setq fill-column 72)
	    (whitespace-mode 1)))

;;;------------------------------------------------------------------------;;;
;;; C coding stuff: indentation, deletion, tabification			   ;;;
;;;------------------------------------------------------------------------;;;
(add-to-list 'auto-mode-alist '("\\.cppm" . c++-mode))

(add-hook 'c-mode-common-hook
	  (function
	   (lambda ()
	     (setq c-offsets-alist
		   (append
		    '(
		      ;; (inextern-lang		. 0)
		      ;; (arglist-cont-nonempty . 4)
		      ;; (arglist-close . c-lineup-arglist-close-under-paren)
		      (case-label	     . *)  ;
		      (statement-case-intro  . *)  ;
		      (substatement-open     . 0)  ; no indent for empty {
		      (statement-cont	     . c-lineup-assignments) ; lines up =
		      (stream-op	     . c-lineup-streamop)
		      (label		     . *)  ; labels are shifted 1/2 step
		      (access-label	     . /)  ;
		      (brace-list-intro	     . +)  ; for array initializers
		      (inlambda		     . 0)  ; lambdas look like functions
		      (innamespace	     . +)  ; from llvm
		      (arglist-intro	     . ++)
		      (arglist-close	     . 0)
		      (member-init-intro     . ++)); from llvm
		    c-offsets-alist))
	     (c-toggle-hungry-state 0)
	     (flyspell-prog-mode)
	     (make-local-variable 'write-contents-hooks))))

;;;------------------------------------------------------------------------;;;
;;; Spell check git commit messages.					   ;;;
;;;------------------------------------------------------------------------;;;
(add-hook 'git-commit-setup-hook 'git-commit-turn-on-flyspell)

;(require 'tex-site)

(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(default ((t (:background "nil")))))
(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(custom-safe-themes
   '("16dd114a84d0aeccc5ad6fd64752a11ea2e841e3853234f19dc02a7b91f5d661" default))
 '(package-selected-packages
   '(use-package lua-mode)))
